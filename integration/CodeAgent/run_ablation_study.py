import argparse
import ast
import json
import os
import re
import shutil
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Force this script to use GPU 0 by default (or CODEAGENT_CUDA_VISIBLE_DEVICES when explicitly set).
os.environ["CUDA_VISIBLE_DEVICES"] = os.getenv("CODEAGENT_CUDA_VISIBLE_DEVICES", "2")

# 
from agents.code_generator import CodeGenerator
from agents.code_retriever import CodeRetriever
from agents.test_agent import TestAgent, get_test_agent
from agents.reflection import ReflectionAgent
from tqdm import tqdm

from app import _load_repo_data
from utils.file_util import load_json
from llm_client import (
    get_deepseekcoder_vllm_client,
    get_gemini_client,
    get_openai_client,
    get_qwen_client,
    get_qwencoder_vllm_client,
)

BASE_DIR = Path(__file__).parent
CONTEXT_ALL_FILENAME = "CONTEXT_ALL.json"
PROMPT_ALL_FILENAME = "PROMPT_ALL.json"
SHARD_DIRNAME = ".tmp_aggregate_shards"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run ablation with repo-level parallelism and repo-scoped checks")
    p.add_argument("--dataset", default="DevEval", help="Dataset name passed to _load_repo_data")
    p.add_argument("--result-load-dir", default=str(BASE_DIR / "results"), help="Where CONTEXT/PROMPT files are loaded")
    p.add_argument("--max-retry", type=int, default=5)
    p.add_argument("--max-repo-workers", type=int, default=1)
    p.add_argument("--generation-model", default="qwencoder")
    p.add_argument("--general-model", default="gpt-4o-mini")
    p.add_argument("--ablation-save-dir", default=None, help="Output directory for ablation results")
    args = p.parse_args()
    model_name = (args.generation_model or "model").strip().lower()
    dataset_name = (args.dataset or "dataset").strip()
    if args.ablation_save_dir:
        args.ablation_save_dir = str(Path(args.ablation_save_dir))
    else:
        args.ablation_save_dir = str(BASE_DIR / f"results_{model_name}" / dataset_name)
    return args


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


def _resolve_sample_id(task_info: dict, dataset: str) -> str:
    if (dataset or "").strip().lower() != "coderevalpython":
        return ""
    for key in ("_idx", "sample_id", "_id", "id", "idx"):
        value = task_info.get(key)
        if value is None:
            continue
        value_str = str(value).strip()
        if value_str:
            return value_str
    return ""


def _resolve_record_key(dataset: str, namespace: str, task_info: Optional[dict] = None, ctx: Optional[dict] = None) -> str:
    if (dataset or "").strip().lower() == "coderevalpython":
        for source in (task_info or {}, ctx or {}):
            for key in ("_idx", "sample_id", "_id", "id", "idx"):
                value = source.get(key)
                if value is None:
                    continue
                text = str(value).strip()
                if text:
                    return text
    return namespace


def _safe_repo_name(repo: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]", "_", repo or "unknown_repo")


def _extract_code_from_response(response_content: str) -> str:
    text = (response_content or "").strip()
    code_match = re.search(r"```(?:[a-zA-Z0-9_+-]+)?\s*\n(.*?)\n```", text, re.DOTALL)
    if code_match:
        return code_match.group(1).strip()

    # Fallback cleanup for malformed/incomplete fenced blocks.
    text = re.sub(r"^\s*```(?:[a-zA-Z0-9_+-]+)?\s*\n?", "", text)
    text = re.sub(r"\n?\s*```\s*$", "", text)
    return text.strip()


def _extract_target_function_code(code_text: str, function_name: str) -> str:
    cleaned = _extract_code_from_response(code_text or "")
    target = (function_name or "").strip()
    if "." in target:
        target = target.split(".")[-1].strip()
    if not target:
        return cleaned

    # Prefer AST extraction for robust function body slicing.
    try:
        tree = ast.parse(cleaned)
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == target:
                segment = ast.get_source_segment(cleaned, node)
                if segment:
                    return segment.strip()
    except Exception:
        pass

    # Fallback: indentation-based extraction from text.
    lines = cleaned.splitlines()
    start_idx = None
    def_indent = 0
    fn_pattern = re.compile(rf"^(\s*)(?:async\s+def|def)\s+{re.escape(target)}\s*\(")
    for i, line in enumerate(lines):
        m = fn_pattern.match(line)
        if m:
            start_idx = i
            def_indent = len(m.group(1))
            break

    if start_idx is None:
        return cleaned

    decorator_start = start_idx
    for j in range(start_idx - 1, -1, -1):
        prev = lines[j]
        if prev.strip().startswith("@") and len(prev) - len(prev.lstrip(" ")) == def_indent:
            decorator_start = j
            continue
        break

    end_idx = len(lines)
    for i in range(start_idx + 1, len(lines)):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            continue
        curr_indent = len(line) - len(line.lstrip(" "))
        if curr_indent <= def_indent:
            end_idx = i
            break

    return "\n".join(lines[decorator_start:end_idx]).strip()


def _invoke_generation_with_saved_prompt(generator: CodeGenerator, prompt_text: str):
    response = generator.llm.invoke(prompt_text)
    response_content = getattr(response, "content", "")
    if isinstance(response_content, list):
        response_content = "\n".join(str(item) for item in response_content)
    response_content = str(response_content)
    print("first gen code:" + response_content)
    return prompt_text, _extract_code_from_response(response_content)


def _format_selected_libraries_for_prompt(selected_libraries: Any) -> str:
    if not selected_libraries:
        return "None"
    if isinstance(selected_libraries, str):
        return selected_libraries.strip() or "None"
    if isinstance(selected_libraries, list):
        lines = []
        for item in selected_libraries:
            if isinstance(item, dict):
                name = str(item.get("name", "")).strip()
                desc = str(item.get("description", "")).strip()
                url = str(item.get("url", "")).strip()
                parts = [p for p in [name, desc, url] if p]
                if parts:
                    lines.append(" | ".join(parts))
            else:
                text = str(item).strip()
                if text:
                    lines.append(text)
        return "\n".join(lines) if lines else "None"
    return str(selected_libraries).strip() or "None"


def _reconstruct_first_prompt_from_context(ctx: dict, task_info: dict) -> str:
    function_name = (task_info.get("function_name") or ctx.get("namespace") or "").strip()
    original_requirement = (
        ctx.get("enhanced_requirement")
        or ctx.get("original_requirement")
        or task_info.get("all_requirement")
        or ""
    ).strip()
    related_code = (ctx.get("selected_snippets_for_gen") or "None").strip() or "None"
    libraries = _format_selected_libraries_for_prompt(ctx.get("selected_libraries"))
    input_code = (task_info.get("input_code") or task_info.get("all_requirement") or "").strip()
    retry_note = "This is the first generation."

    return f"""
You are a professional AI code generation assistant. Your task is to generate high-quality Python code for a single function.

Follow these instructions carefully:

---

### 1. Implementation Scope
- Only implement the function {function_name} as provided in the Input Code.
- The implementation must strictly follow the original requirement, which is described as follows:
  - Original Requirement: 
  {original_requirement}
- Do not modify or generate any code outside this function (no extra classes, functions, imports, or variables).
- Do not add extra imports or helper functions outside this function unless absolutely necessary. Use only the provided function signature.
- Follow exactly the parameter names, order, and logic structure in the input code.

---

### 2. References: Libraries
- You may use the following third-party libraries if needed:
- The following are the reference code third-party libraries:
{libraries}

---

### 3. References: Similar Code Snippets
- The retrieved code snippets illustrate patterns and approaches for solving similar tasks.
- Use them as high-level guidance only—follow the design patterns or approaches, but do not copy code verbatim or change the input function signature or parameters.
- The following are the reference code snippets:
{related_code}

---

### 4. Input Code
- Use the following input code as the base. Implement only the specified function.
Input Code:
{input_code}

---

### 5. Retry Note
{retry_note}

---

### 6. Output Format
- Return only the completed Python function code. Do not include any explanations, markdown, or extra text outside the code block.
- Wrap the code exactly with triple backticks and specify Python:

```python
# Generated code
```
""".strip()


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


def _resolve_effective_load_dir(result_load_dir: Path, dataset: str) -> Path:
    # ： <result-load-dir>/<dataset>
    # （basename==dataset），
    if result_load_dir.name == dataset:
        return result_load_dir

    nested = result_load_dir / dataset
    if nested.exists():
        return nested

    raise FileNotFoundError(
        f"Dataset-specific load dir not found: {nested}. "
        f"Please provide --result-load-dir as parent dir containing '{dataset}/' or as the dataset dir itself."
    )


def _collect_context_records(load_dir: Path) -> List[Tuple[str, dict]]:
    context_all_path = load_dir / CONTEXT_ALL_FILENAME
    if context_all_path.exists():
        context_index = _load_json_dict(context_all_path)
        records: List[Tuple[str, dict]] = []
        for key, value in context_index.items():
            if isinstance(value, dict):
                records.append((str(key), value))
        return records

    records = []
    for ctx_path in sorted(load_dir.glob("CONTEXT_*.json")):
        try:
            ctx = load_json(ctx_path)
            if not isinstance(ctx, dict):
                continue
            key = ctx_path.stem.replace("CONTEXT_", "", 1)
            records.append((key, ctx))
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


def _extract_prompt_from_index(prompt_index: Dict[str, Any], candidates: List[str]) -> str:
    for key in candidates:
        if not key:
            continue
        obj = prompt_index.get(key)
        if isinstance(obj, dict):
            val = (obj.get("first_gen_prompt") or "").strip()
            if val:
                return val
        elif isinstance(obj, str):
            val = obj.strip()
            if val:
                return val
    return ""


def _build_agents(generation_model: str, general_model: str, dataset: str):
    generator = CodeGenerator(model_name=general_model)
    model_name = (generation_model or "").strip().lower()
    generation_temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    generation_top_p = float(os.getenv("LLM_TOP_P", "1.0"))
    if model_name in {"deepseekcoder", "deepseek-coder-6.7b"}:
        generator.llm = get_deepseekcoder_vllm_client(model="deepseek-coder-6.7b")
    elif model_name in {"qwencoder", "qwencoder-2.5-7b"}:
        generator.llm = get_qwencoder_vllm_client(model="qwencoder-2.5-7b")
    elif model_name.startswith("gemini"):
        gemini_model = generation_model if generation_model and generation_model != "gemini" else "gemini-2.5-flash"
        generator.llm = get_gemini_client(model=gemini_model)
    elif model_name.startswith("gpt-") or model_name.startswith("o"):
        openai_model = generation_model if generation_model else "gpt-4o-mini"
        generator.llm = get_openai_client(
            model=openai_model,
            temperature=generation_temperature,
            top_p=generation_top_p,
        )
    else:
        generator.llm = get_qwen_client(model=generation_model)
    retriever = CodeRetriever(dataset=dataset)
    test_agent = get_test_agent(dataset=dataset, model_name=general_model)
    ref_agent = ReflectionAgent(generator, retriever, model_name=general_model)
    return generator, test_agent, ref_agent


def _preload_generation_llm(generation_model: str):
    model_name = (generation_model or "").strip().lower()
    print(f"[Warmup] Preloading generation model: {generation_model}")
    generation_temperature = float(os.getenv("LLM_TEMPERATURE", "0"))
    generation_top_p = float(os.getenv("LLM_TOP_P", "1.0"))
    if model_name in {"deepseekcoder", "deepseek-coder-6.7b"}:
        get_deepseekcoder_vllm_client(model="deepseek-coder-6.7b")
    elif model_name in {"qwencoder", "qwencoder-2.5-7b"}:
        get_qwencoder_vllm_client(model="qwencoder-2.5-7b")
    elif model_name.startswith("gemini"):
        gemini_model = generation_model if generation_model and generation_model != "gemini" else "gemini-2.5-flash"
        get_gemini_client(model=gemini_model)
    elif model_name.startswith("gpt-") or model_name.startswith("o"):
        openai_model = generation_model if generation_model else "gpt-4o-mini"
        get_openai_client(
            model=openai_model,
            temperature=generation_temperature,
            top_p=generation_top_p,
        )
    else:
        get_qwen_client(model=generation_model)
    print("[Warmup] Generation model is ready.")


def _repo_key(ctx: dict, task_info: dict) -> str:
    return (
        ctx.get("repo_name")
        or task_info.get("repo_name")
        or task_info.get("project_path")
        or "__unknown_repo__"
    )


def _process_one_context(
    record_key: str,
    ctx: dict,
    task_info: dict,
    generator: CodeGenerator,
    test_agent: TestAgent,
    ref_agent: ReflectionAgent,
    repo_data_cache: dict,
    args: argparse.Namespace,
    func_path: Path,
    load_dir: Path,
    prompt_index: Dict[str, Any],
    existing_ablation_keys: set,
    context_shard: Dict[str, Any],
):
    namespace = ctx.get("namespace") or task_info.get("namespace")
    if not namespace:
        return "skipped"

    task_key = _resolve_record_key(args.dataset, namespace, task_info=task_info, ctx=ctx)
    if task_key in existing_ablation_keys:
        return "skipped"

    requirement = ctx.get("original_requirement", "")
    repo_name = ctx.get("repo_name", "")
    function_name = (task_info.get("function_name") or namespace.split(".")[-1]).strip()
    project_path = task_info.get("project_path", "")
    enhanced_req = ctx.get("enhanced_requirement", requirement)

    if repo_name not in repo_data_cache:
        repo_data_cache[repo_name] = _load_repo_data(repo_name, args.dataset)
    repo_data = repo_data_cache[repo_name]

    print(f"\n{'='*40}\n[ABLATION] Processing: {namespace}\n{'='*40}")

    fname = namespace.replace(".", "_")
    prompt_candidates = [task_key, namespace, fname]
    saved_prompt = _extract_prompt_from_index(prompt_index, prompt_candidates)

    if not saved_prompt:
        legacy_prompt_path = load_dir / f"PROMPT_{fname}.json"
        if legacy_prompt_path.exists():
            prompt_obj = load_json(legacy_prompt_path)
            saved_prompt = (prompt_obj.get("first_gen_prompt") or "").strip()

    if not saved_prompt:
        print(f"  [Warn] Missing prompt, reconstruct from CONTEXT: {namespace}")
        saved_prompt = _reconstruct_first_prompt_from_context(ctx, task_info)

    if not saved_prompt:
        raise ValueError(f"Failed to reconstruct first prompt for {namespace}")

    print("  [Step] Initial Generation with saved prompt")
    _, first_code = _invoke_generation_with_saved_prompt(generator, saved_prompt)
    curr_code = _extract_target_function_code(first_code, function_name)

    test_dir = task_info["repo_dir_path"].split("\\")
    test_codes = []
    test_file_path = os.path.join(func_path, test_dir[-2], test_dir[-1], "tests.jsonl")
    if os.path.exists(test_file_path):
        with open(test_file_path, "r", encoding="utf-8") as tf:
            for line in tf:
                test_codes.append(json.loads(line))
    sample_id = _resolve_sample_id(task_info, args.dataset)

    ablation_test_history = []
    ablation_gen_history = []

    retry_count = 0
    while retry_count <= args.max_retry:
        print(f"  [Test] Attempt {retry_count}...")
        t_res = test_agent.test(namespace, curr_code, test_codes, project_path, sample_id=sample_id)

        ablation_test_history.append(
            {"retry": retry_count, "passed": t_res.passed, "error": t_res.error_message}
        )
        ablation_gen_history.append({"retry": retry_count, "code": curr_code})

        if t_res.passed:
            print(f"  ✓ Passed at retry {retry_count}!")
            break

        if retry_count < args.max_retry:
            retry_count += 1
            print(f"  [Reflect] Regenerating for attempt {retry_count}...")
            reflection = ref_agent.reflect_and_regenerate(
                data={**task_info, "dataset": args.dataset},
                function_name=function_name,
                input_code=task_info["input_code"],
                class_definitions=enhanced_req,
                test_result=t_res,
                current_code=curr_code,
                retry_count=retry_count,
                repo_data=repo_data,
            )
            curr_code = _extract_target_function_code(reflection.code, function_name)
        else:
            print(f"  ❌ Failed after {args.max_retry} retries.")
            break

    context_shard[task_key] = {
        "record_key": task_key,
        "namespace": namespace,
        "test_history": ablation_test_history,
        "generated_history": ablation_gen_history,
        "final_code": curr_code,
    }
    return "done"


def _run_repo_group(
    repo: str,
    items: List[Tuple[str, dict, dict]],
    args: argparse.Namespace,
    func_path: Path,
    load_dir: Path,
    prompt_index: Dict[str, Any],
    existing_ablation_keys: set,
    shard_dir: Path,
):
    print(f"[RepoWorker] Start repo={repo}, tasks={len(items)}")
    generator, test_agent, ref_agent = _build_agents(args.generation_model, args.general_model, args.dataset)
    repo_data_cache: dict = {}

    done = 0
    skipped = 0
    errors = 0
    context_shard: Dict[str, Any] = {}

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
                args=args,
                func_path=func_path,
                load_dir=load_dir,
                prompt_index=prompt_index,
                existing_ablation_keys=existing_ablation_keys,
                context_shard=context_shard,
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
    # Merge with pre-existing shard content to support resume without data loss.
    existing_context = _load_json_dict(context_shard_path)
    merged_context = dict(existing_context)
    merged_context.update(context_shard)
    _atomic_dump_json(context_shard_path, merged_context)

    print(f"[RepoWorker] Done repo={repo}, done={done}, skipped={skipped}, errors={errors}")
    return {
        "repo": repo,
        "done": done,
        "skipped": skipped,
        "errors": errors,
        "context_shard": str(context_shard_path),
    }


def run_strict_ablation(args: argparse.Namespace):
    result_load_dir = Path(args.result_load_dir)
    load_dir = _resolve_effective_load_dir(result_load_dir, args.dataset)
    ablation_save_dir = Path(args.ablation_save_dir)
    func_path = _resolve_func_path(args.dataset)
    print(f"Using func path: {func_path}")
    print(f"Using result load dir: {load_dir}")

    ablation_save_dir.mkdir(parents=True, exist_ok=True)
    shard_dir = ablation_save_dir / SHARD_DIRNAME
    shard_dir.mkdir(parents=True, exist_ok=True)
    _preload_generation_llm(args.generation_model)

    context_records = _collect_context_records(load_dir)
    if not context_records:
        print("No CONTEXT records found to run.")
        return

    dataset_json_path = _resolve_dataset_json_path(args.dataset)
    print(f"Using dataset json: {dataset_json_path}")
    data = load_json(dataset_json_path)

    data_by_namespace: Dict[str, dict] = {}
    data_by_idx: Dict[str, dict] = {}
    for d in data:
        ns = d.get("namespace")
        if ns:
            data_by_namespace[ns] = d
        idx_key = _resolve_record_key(args.dataset, ns or "", task_info=d)
        if idx_key:
            data_by_idx[idx_key] = d

    ablation_context_path = ablation_save_dir / CONTEXT_ALL_FILENAME
    ablation_context_index = _load_json_dict(ablation_context_path)
    shard_existing_keys = _collect_existing_keys_from_context_shards(shard_dir)
    existing_ablation_keys = set(ablation_context_index.keys()) | shard_existing_keys
    if shard_existing_keys:
        print(f"Detected {len(shard_existing_keys)} existing keys from shard dir: {shard_dir}")

    prompt_all_path = load_dir / PROMPT_ALL_FILENAME
    prompt_index = _load_json_dict(prompt_all_path)

    repo_groups: Dict[str, List[Tuple[str, dict, dict]]] = defaultdict(list)
    unmatched = 0

    for record_key, ctx in context_records:
        try:
            namespace = ctx.get("namespace", "")
            task_info = data_by_namespace.get(namespace)
            if not task_info:
                task_info = data_by_idx.get(record_key) or data_by_idx.get(
                    _resolve_record_key(args.dataset, namespace, ctx=ctx)
                )
            if not task_info:
                unmatched += 1
                print(f"[Warn] task not found in dataset, skip: key={record_key}, namespace={namespace}")
                continue

            repo = _repo_key(ctx, task_info)
            repo_groups[repo].append((record_key, ctx, task_info))
        except Exception as e:
            print(f"❌ Failed to prepare task key={record_key}: {e}")
            traceback.print_exc()

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

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _run_repo_group,
                repo,
                items,
                args,
                func_path,
                load_dir,
                prompt_index,
                existing_ablation_keys,
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

    #  context 
    merged_context = dict(ablation_context_index)
    for shard in context_shards:
        shard_data = _load_json_dict(shard)
        merged_context.update(shard_data)
    _atomic_dump_json(ablation_context_path, merged_context)

    # 
    shutil.rmtree(shard_dir, ignore_errors=True)

    print(
        "All done. "
        f"done={totals['done']}, skipped={totals['skipped']}, errors={totals['errors']}, "
        f"repos={len(repo_groups)}, merged_context={ablation_context_path}"
    )


if __name__ == "__main__":
    run_strict_ablation(parse_args())
