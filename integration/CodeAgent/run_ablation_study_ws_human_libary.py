import json
import os
import traceback
from pathlib import Path
from typing import Any, Dict, List, Tuple

# 
from agents.code_generator import CodeGenerator
from agents.code_retriever import CodeRetriever
from agents.test_agent import TestAgent
from agents.reflection import ReflectionAgent
from tqdm import tqdm

from app import _load_repo_data
from utils.file_util import load_json

# ---  ---
BASE_DIR = Path(__file__).parent
DATASET = "DevEval"
CONTEXT_ALL_FILENAME = "CONTEXT_ALL.json"
PROMPT_ALL_FILENAME = "PROMPT_ALL.json"

RESULT_LOAD_DIR = BASE_DIR / "results"
ABLATION_SAVE_DIR = BASE_DIR / "results_ws_human_in_select_libraries"
ABLATION_SAVE_DIR.mkdir(exist_ok=True)

FUNC_PATH = Path("")
MAX_RETRY = 5


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


def get_all_libs(all_recommended_libraries):
    all_libs = []
    for task_group in all_recommended_libraries or []:
        libraries = task_group.get("libraries", [])
        for lib in libraries:
            all_libs.append(
                {
                    "name": lib.get("name"),
                    "description": lib.get("description"),
                    "url": lib.get("url"),
                }
            )
    return all_libs


def run_strict_ablation():
    generator = CodeGenerator()
    retriever = CodeRetriever()
    test_agent = TestAgent()
    ref_agent = ReflectionAgent(generator, retriever)

    load_dir = _resolve_effective_load_dir(RESULT_LOAD_DIR, DATASET)
    context_records = _collect_context_records(load_dir)

    out_context_path = ABLATION_SAVE_DIR / CONTEXT_ALL_FILENAME
    out_prompt_path = ABLATION_SAVE_DIR / PROMPT_ALL_FILENAME
    out_context_index = _load_json_dict(out_context_path)
    out_prompt_index = _load_json_dict(out_prompt_path)

    data = load_json("")
    data_by_namespace = {d.get("namespace"): d for d in data if d.get("namespace")}

    for record_key, ctx in tqdm(context_records):
        try:
            namespace = ctx["namespace"]
            if record_key in out_context_index:
                print(f"  [Skip] Result exists: key={record_key}")
                continue

            task_info = data_by_namespace.get(namespace)
            if not task_info:
                raise KeyError(f"task_info not found for namespace: {namespace}")

            requirement = ctx.get("original_requirement", "")
            repo_name = ctx.get("repo_name", "")
            project_path = ctx.get("project_path") or task_info.get("project_path")
            enhanced_req = ctx.get("enhanced_requirement") or requirement
            repo_data = _load_repo_data(repo_name, DATASET)

            print(f"\n{'='*40}\n[ABLATION] Processing: {namespace}\n{'='*40}")

            prompt_text, generated = generator.generate(
                function_name=namespace.split(".")[-1],
                input_code=requirement,
                original_requirement=enhanced_req,
                related_code=ctx.get("selected_snippets_for_gen"),
                selected_libraries=get_all_libs(ctx.get("all_recommended_libraries", [])),
                retries=0,
            )
            curr_code = generated.code

            test_dir = task_info["repo_dir_path"].split("\\")
            test_codes = []
            test_file_path = os.path.join(FUNC_PATH, test_dir[-2], test_dir[-1], "tests.jsonl")
            if os.path.exists(test_file_path):
                with open(test_file_path, "r", encoding="utf-8") as tf:
                    for line in tf:
                        test_codes.append(json.loads(line))

            ablation_test_history = []
            ablation_gen_history = []

            retry_count = 0
            while retry_count <= MAX_RETRY:
                print(f"  [Test] Attempt {retry_count}...")
                t_res = test_agent.test(namespace, curr_code, test_codes, project_path)

                ablation_test_history.append({"retry": retry_count, "passed": t_res.passed, "error": t_res.error_message})
                ablation_gen_history.append({"retry": retry_count, "code": curr_code})

                if t_res.passed:
                    print(f"  ✓ Passed at retry {retry_count}!")
                    break

                if retry_count < MAX_RETRY:
                    retry_count += 1
                    print(f"  [Reflect] Regenerating for attempt {retry_count}...")
                    reflection = ref_agent.reflect_and_regenerate(
                        data=task_info,
                        function_name=repo_name,
                        input_code=task_info["input_code"],
                        class_definitions=enhanced_req,
                        test_result=t_res,
                        current_code=curr_code,
                        retry_count=retry_count,
                        repo_data=repo_data,
                    )
                    curr_code = reflection.code
                else:
                    print(f"  ❌ Failed after {MAX_RETRY} retries.")
                    break

            out_context_index[record_key] = {
                "record_key": record_key,
                "namespace": namespace,
                "test_history": ablation_test_history,
                "generated_history": ablation_gen_history,
                "final_code": curr_code,
            }
            out_prompt_index[record_key] = {
                "record_key": record_key,
                "namespace": namespace,
                "first_gen_prompt": prompt_text,
            }

        except Exception as e:
            print(f"❌ Error in key={record_key}: {e}")
            traceback.print_exc()

    _atomic_dump_json(out_context_path, out_context_index)
    _atomic_dump_json(out_prompt_path, out_prompt_index)


if __name__ == "__main__":
    run_strict_ablation()
