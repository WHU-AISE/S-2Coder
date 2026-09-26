import argparse
import json
import os
import re
import shutil
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

from agents.code_generator import CodeGenerator
from agents.code_retriever import CodeRetriever
from agents.test_agent import TestAgent
from agents.reflection import ReflectionAgent
from tqdm import tqdm

from app import _load_repo_data
from utils.file_util import load_json

BASE_DIR = Path(__file__).parent
CONTEXT_ALL_FILENAME = "CONTEXT_ALL.json"
PROMPT_ALL_FILENAME = "PROMPT_ALL.json"
SHARD_DIRNAME = ".tmp_aggregate_shards"

NO_THIRD_PARTY_LIBRARIES = (
    "Do not use any third-party libraries. "
    "Only Python standard library modules are allowed when absolutely necessary."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run ablation without third-party libraries, with repo-level parallelism"
    )
    p.add_argument("--dataset", default="DevEval")
    p.add_argument("--result-load-dir", default=str(BASE_DIR / "results"))
    p.add_argument("--max-retry", type=int, default=5)
    p.add_argument("--max-repo-workers", type=int, default=1)
    p.add_argument(
        "--ablation-save-dir",
        default=str(BASE_DIR / "results_wo_third_party_library"),
        help="Output directory for ablation results",
    )
    return p.parse_args()


def _resolve_dataset_json_path(dataset: str) -> Path:
    dataset_dir = BASE_DIR / "dataset"
    exact = dataset_dir / f"{dataset}_dataset_all_infor_172.json"
    if exact.exists():
        return exact

    candidates = sorted(dataset_dir.glob(f"{dataset}_dataset_all_infor_*.json"))
    if candidates:
        return candidates[-1]

    fallback = dataset_dir / f"{dataset}.json"
    if fallback.exists():
        return fallback

    raise FileNotFoundError(f"Cannot resolve dataset json for dataset={dataset} in {dataset_dir}")


def _resolve_func_path(dataset: str) -> Path:
    return Path("")


def _resolve_effective_load_dir(base_dir: Path, dataset: str) -> Path:
    direct = base_dir / CONTEXT_ALL_FILENAME
    if direct.exists():
        return base_dir
    nested = base_dir / dataset
    if (nested / CONTEXT_ALL_FILENAME).exists() or nested.exists():
        return nested
    return base_dir


def _load_json_dict(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = load_json(path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _atomic_dump_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    tmp.replace(path)


def _collect_context_records(load_dir: Path) -> List[Tuple[str, dict]]:
    context_all = load_dir / CONTEXT_ALL_FILENAME
    if context_all.exists():
        data = _load_json_dict(context_all)
        return [(str(k), v) for k, v in data.items() if isinstance(v, dict)]

    records = []
    for p in sorted(load_dir.glob("CONTEXT_*.json")):
        try:
            obj = load_json(p)
            if isinstance(obj, dict):
                key = p.stem.replace("CONTEXT_", "", 1)
                records.append((key, obj))
        except Exception:
            continue
    return records


def _collect_existing_keys_from_context_shards(shard_dir: Path) -> set:
    keys = set()
    if not shard_dir.exists():
        return keys
    for shard_path in sorted(shard_dir.glob("context_*.json")):
        shard_data = _load_json_dict(shard_path)
        keys.update(str(k) for k in shard_data.keys())
    return keys


def _safe_repo_name(repo: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", repo or "unknown_repo")


def _repo_key(ctx: dict, task_info: dict) -> str:
    return (
        ctx.get("repo_name")
        or task_info.get("repo_name")
        or task_info.get("project_path")
        or "__unknown_repo__"
    )


def _attach_no_third_party_constraint(text: str) -> str:
    base = (text or "").strip()
    if NO_THIRD_PARTY_LIBRARIES in base:
        return base
    if base:
        return base + "\n\nConstraint: " + NO_THIRD_PARTY_LIBRARIES
    return "Constraint: " + NO_THIRD_PARTY_LIBRARIES


def _process_one_context(
    record_key: str,
    ctx: dict,
    task_info: dict,
    generator: CodeGenerator,
    test_agent: TestAgent,
    ref_agent: ReflectionAgent,
    repo_data_cache: dict,
    dataset: str,
    func_path: Path,
    max_retry: int,
    existing_keys: set,
    context_shard: Dict[str, Any],
    prompt_shard: Dict[str, Any],
):
    if record_key in existing_keys:
        return "skipped"

    namespace = ctx.get("namespace")
    if not namespace:
        return "skipped"

    requirement = ctx.get("original_requirement", "")
    repo_name = ctx.get("repo_name", "")
    project_path = ctx.get("project_path") or task_info.get("project_path")
    enhanced_req = ctx.get("enhanced_requirement") or requirement
    constrained_requirement = _attach_no_third_party_constraint(enhanced_req)

    if repo_name not in repo_data_cache:
        repo_data_cache[repo_name] = _load_repo_data(repo_name, dataset)
    repo_data = repo_data_cache[repo_name]

    print(f"\n{'='*40}\n[ABLATION] Processing: {namespace}\n{'='*40}")

    prompt_text, generated = generator.generate(
        function_name=namespace.split(".")[-1],
        input_code=requirement,
        original_requirement=constrained_requirement,
        related_code=ctx.get("selected_snippets_for_gen"),
        # selected_libraries=NO_THIRD_PARTY_LIBRARIES,
        retries=0,
    )
    curr_code = generated.code

    test_dir = task_info["repo_dir_path"].split("\\")
    test_codes = []
    test_file_path = os.path.join(func_path, test_dir[-2], test_dir[-1], "tests.jsonl")
    if os.path.exists(test_file_path):
        with open(test_file_path, "r", encoding="utf-8") as tf:
            for line in tf:
                test_codes.append(json.loads(line))

    ablation_test_history = []
    ablation_gen_history = []

    retry_count = 0
    while retry_count <= max_retry:
        print(f"  [Test] Attempt {retry_count}...")
        t_res = test_agent.test(namespace, curr_code, test_codes, project_path)

        ablation_test_history.append({"retry": retry_count, "passed": t_res.passed, "error": t_res.error_message})
        ablation_gen_history.append({"retry": retry_count, "code": curr_code})

        if t_res.passed:
            print(f"  ✓ Passed at retry {retry_count}!")
            break

        if retry_count < max_retry:
            retry_count += 1
            print(f"  [Reflect] Regenerating for attempt {retry_count}...")
            reflection = ref_agent.reflect_and_regenerate(
                data={**task_info, "dataset": dataset},
                function_name=repo_name,
                input_code=task_info["input_code"],
                class_definitions=constrained_requirement,
                test_result=t_res,
                current_code=curr_code,
                retry_count=retry_count,
                repo_data=repo_data,
            )
            curr_code = reflection.code
        else:
            print(f"  ❌ Failed after {max_retry} retries.")
            break

    context_shard[record_key] = {
        "record_key": record_key,
        "namespace": namespace,
        "test_history": ablation_test_history,
        "generated_history": ablation_gen_history,
        "final_code": curr_code,
    }
    prompt_shard[record_key] = {
        "record_key": record_key,
        "namespace": namespace,
        "first_gen_prompt": prompt_text,
    }
    return "done"


def _run_repo_group(
    repo: str,
    items: List[Tuple[str, dict, dict]],
    dataset: str,
    func_path: Path,
    max_retry: int,
    existing_keys: set,
    shard_dir: Path,
):
    print(f"[RepoWorker] Start repo={repo}, tasks={len(items)}")
    generator = CodeGenerator()
    retriever = CodeRetriever(dataset=dataset)
    test_agent = TestAgent()
    ref_agent = ReflectionAgent(generator, retriever)
    repo_data_cache: dict = {}

    done = 0
    skipped = 0
    errors = 0
    context_shard: Dict[str, Any] = {}
    prompt_shard: Dict[str, Any] = {}

    for record_key, ctx, task_info in items:
        try:
            status = _process_one_context(
                record_key=record_key,
                ctx=ctx,
                task_info=task_info,
                generator=generator,
                test_agent=test_agent,
                ref_agent=ref_agent,
                repo_data_cache=repo_data_cache,
                dataset=dataset,
                func_path=func_path,
                max_retry=max_retry,
                existing_keys=existing_keys,
                context_shard=context_shard,
                prompt_shard=prompt_shard,
            )
            if status == "done":
                done += 1
            elif status == "skipped":
                skipped += 1
        except Exception as e:
            errors += 1
            name = ctx.get("namespace") or record_key
            print(f"❌ Error in {name}: {e}")
            traceback.print_exc()

    safe_repo = _safe_repo_name(repo)

    context_shard_path = shard_dir / f"context_{safe_repo}.json"
    existing_context = _load_json_dict(context_shard_path)
    merged_context = dict(existing_context)
    merged_context.update(context_shard)
    _atomic_dump_json(context_shard_path, merged_context)

    prompt_shard_path = shard_dir / f"prompt_{safe_repo}.json"
    existing_prompt = _load_json_dict(prompt_shard_path)
    merged_prompt = dict(existing_prompt)
    merged_prompt.update(prompt_shard)
    _atomic_dump_json(prompt_shard_path, merged_prompt)

    print(f"[RepoWorker] Done repo={repo}, done={done}, skipped={skipped}, errors={errors}")
    return {
        "repo": repo,
        "done": done,
        "skipped": skipped,
        "errors": errors,
        "context_shard": str(context_shard_path),
        "prompt_shard": str(prompt_shard_path),
    }


def run_strict_ablation_wo_third_party_library(args: argparse.Namespace):
    load_dir = _resolve_effective_load_dir(Path(args.result_load_dir), args.dataset)
    save_dir = Path(args.ablation_save_dir)
    func_path = _resolve_func_path(args.dataset)

    print(f"Using func path: {func_path}")
    print(f"Using result load dir: {load_dir}")

    save_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = save_dir / SHARD_DIRNAME
    shard_dir.mkdir(parents=True, exist_ok=True)

    context_records = _collect_context_records(load_dir)
    if not context_records:
        print("No CONTEXT records found to run.")
        return

    dataset_json_path = _resolve_dataset_json_path(args.dataset)
    print(f"Using dataset json: {dataset_json_path}")
    data = load_json(dataset_json_path)
    data_by_namespace = {d.get("namespace"): d for d in data if d.get("namespace")}

    out_context_path = save_dir / CONTEXT_ALL_FILENAME
    out_prompt_path = save_dir / PROMPT_ALL_FILENAME
    out_context_index = _load_json_dict(out_context_path)
    out_prompt_index = _load_json_dict(out_prompt_path)

    shard_existing_keys = _collect_existing_keys_from_context_shards(shard_dir)
    existing_keys = set(out_context_index.keys()) | shard_existing_keys

    repo_groups: Dict[str, List[Tuple[str, dict, dict]]] = defaultdict(list)
    unmatched = 0

    for record_key, ctx in context_records:
        namespace = ctx.get("namespace", "")
        task_info = data_by_namespace.get(namespace)
        if not task_info:
            unmatched += 1
            print(f"[Warn] task not found in dataset, skip: key={record_key}, namespace={namespace}")
            continue
        repo = _repo_key(ctx, task_info)
        repo_groups[repo].append((record_key, ctx, task_info))

    if not repo_groups:
        print("No repo tasks to run.")
        return

    max_workers = min(max(1, int(args.max_repo_workers)), len(repo_groups))
    print(
        f"Prepared {sum(len(v) for v in repo_groups.values())} tasks across {len(repo_groups)} repos, "
        f"max_repo_workers={max_workers}, unmatched={unmatched}"
    )

    totals = {"done": 0, "skipped": 0, "errors": 0}
    context_shards: List[Path] = sorted(shard_dir.glob("context_*.json"))
    prompt_shards: List[Path] = sorted(shard_dir.glob("prompt_*.json"))

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_repo_group,
                repo,
                items,
                args.dataset,
                func_path,
                args.max_retry,
                existing_keys,
                shard_dir,
            ): repo
            for repo, items in repo_groups.items()
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Repo groups"):
            result = future.result()
            totals["done"] += result["done"]
            totals["skipped"] += result["skipped"]
            totals["errors"] += result["errors"]
            context_shards.append(Path(result["context_shard"]))
            prompt_shards.append(Path(result["prompt_shard"]))

    merged_context = dict(out_context_index)
    for shard in context_shards:
        merged_context.update(_load_json_dict(shard))
    _atomic_dump_json(out_context_path, merged_context)

    merged_prompt = dict(out_prompt_index)
    for shard in prompt_shards:
        merged_prompt.update(_load_json_dict(shard))
    _atomic_dump_json(out_prompt_path, merged_prompt)

    shutil.rmtree(shard_dir, ignore_errors=True)

    print(
        "All done. "
        f"done={totals['done']}, skipped={totals['skipped']}, errors={totals['errors']}, "
        f"repos={len(repo_groups)}, merged_context={out_context_path}, merged_prompt={out_prompt_path}"
    )


if __name__ == "__main__":
    run_strict_ablation_wo_third_party_library(parse_args())
