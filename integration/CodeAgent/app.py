import os
import re
import json
import time
import traceback
import threading
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, List, Any

from flask import Flask, jsonify, request
from flask_socketio import SocketIO, emit
from flask_swagger_ui import get_swaggerui_blueprint

#  Agent 
from agents.search import SearchAgent
from agents.code_generator import CodeGenerator
from agents.code_retriever import CodeRetriever
from agents.dependency_analyzer import DependencyAnalyzer
from agents.reflection import ReflectionAgent
from agents.test_agent import get_test_agent
from agents.requirement_decomposer import RequirementDecomposer, Task
from agents.requirement_reflection import RequirementReflector
from utils.file_util import load_json, load_jsonl, load_pkl_gz, json_to_dict
from utils.time import StageTimer


app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ---  ---
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
DEFAULT_DATASET = "DevEval"
DEFAULT_MODEL = "gpt-4o-mini"
FUNCTIONS_ROOT = Path("")
GRAPHS_ROOT = Path("")
COMMENT_LIBRARY_ROOT = Path("")
DATASET_META_DIR = Path(__file__).parent / "dataset"
RESULT_SAVE_ROOT = Path(__file__).parent / "results"
RESULT_SAVE_ROOT.mkdir(exist_ok=True)
CONTEXT_ALL_FILENAME = "CONTEXT_ALL.json"
PROMPT_ALL_FILENAME = "PROMPT_ALL.json"
RUN_CONFIG = {"model_name": DEFAULT_MODEL}

# ---  ---
task_confirmed_event = threading.Event()
snippets_confirmed_event = threading.Event()

current_ui_feedback = {
    "approved_tasks": None,
    "selected_snippets": None,
    "selected_libraries": None
}

MAX_RETRY = 5


def get_functions_dataset_path(dataset: str) -> Path:
    return FUNCTIONS_ROOT / dataset


def get_graphs_dataset_path(dataset: str) -> Path:
    return GRAPHS_ROOT / dataset


def get_comment_library_path(dataset: str) -> Path:
    return COMMENT_LIBRARY_ROOT / dataset / "Comment_Library"


def get_result_save_dir(dataset: str) -> Path:
    save_dir = RESULT_SAVE_ROOT / dataset
    save_dir.mkdir(parents=True, exist_ok=True)
    return save_dir


def _context_all_path(result_save_dir: Path) -> Path:
    return result_save_dir / CONTEXT_ALL_FILENAME


def _prompt_all_path(result_save_dir: Path) -> Path:
    return result_save_dir / PROMPT_ALL_FILENAME


def _load_json_index(path: Path) -> Dict[str, Any]:
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


def _resolve_record_key(dataset: str, task_info: Dict[str, Any], namespace: str) -> str:
    if (dataset or "").strip().lower() == "coderevalpython":
        for k in ("_idx", "_id", "id", "idx", "sample_id"):
            v = task_info.get(k)
            if v is None:
                continue
            text = str(v).strip()
            if text:
                return text
    return namespace


def resolve_dataset_json_file(dataset: str) -> Optional[Path]:
    exact = DATASET_META_DIR / f"{dataset}_dataset_all_infor_172.json"
    if exact.exists():
        return exact

    candidates = list(DATASET_META_DIR.glob(f"{dataset}_dataset_all_infor_*.json"))
    if not candidates:
        fallback = DATASET_META_DIR / f"{dataset}.json"
        return fallback if fallback.exists() else None

    def _score(p: Path) -> int:
        m = re.search(r"_dataset_all_infor_(\d+)$", p.stem)
        return int(m.group(1)) if m else -1

    candidates.sort(key=_score, reverse=True)
    return candidates[0]

# Swagger 
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL, API_URL, config={'app_name': "Interactive Batch Code Gen API"}
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

@app.route('/')
def index():
    return "Interactive Batch Agent is running. Post to /run_test to start."


# ==========================================
# 1. ：
# ==========================================
@app.route('/run_test', methods=['POST'])
def run_test():
    try:
        dataset = (request.form.get('dataset') or DEFAULT_DATASET).strip()
        model_name = (request.form.get('model') or DEFAULT_MODEL).strip()
        if not dataset:
            return jsonify({'error': 'dataset is required'}), 400
        if not model_name:
            return jsonify({'error': 'model is required'}), 400

        json_file_path = resolve_dataset_json_file(dataset)
        if json_file_path is None:
            return jsonify({'error': f'No dataset json found for {dataset} in {DATASET_META_DIR}'}), 400

        data = load_json(json_file_path)

        if not data:
            return jsonify({'error': 'Dataset is empty'}), 400

        RUN_CONFIG["model_name"] = model_name
        threading.Thread(target=main_batch_loop, args=(data, dataset, model_name), daemon=True).start()
        
        return jsonify({
            'message': f'Started interactive batch for {len(data)} records.',
            'dataset': dataset,
            'model': model_name,
            'dataset_file': str(json_file_path),
            'result_dir': str(get_result_save_dir(dataset))
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


# ==========================================
# 2. 
# ==========================================
def main_batch_loop(data_list, dataset: str, model_name: str):
    result_save_dir = get_result_save_dir(dataset)
    context_path = _context_all_path(result_save_dir)
    context_index = _load_json_index(context_path)

    for idx, task_info in enumerate(data_list):
        namespace = task_info.get("namespace", f"unknown_{idx}")
        record_key = _resolve_record_key(dataset, task_info, namespace)

        if record_key in context_index:
            print(f"[BATCH] Skipping {namespace} (key={record_key}), already processed.")
            continue

        print(f"\n{'='*30}\n[BATCH] Processing {idx+1}/{len(data_list)}: {namespace}\n{'='*30}")
        context = {
            "task_info": task_info,
            "namespace": namespace,
            "record_key": record_key,
            "repo_name": task_info.get("repo_name"),
            "dataset": dataset,
            "model_name": model_name,
            "result_save_dir": result_save_dir,
            "divided_tasks": None,
            "reflection_result": None,
            "task_code_snippets": None,
            "generated_history": [],
            "test_history": [],
            "retry_count": 0,
            "timing": {}
        }

        try:
            run_pipeline_for_single_record(context)
            save_final_results(context)
            context_index[record_key] = {"saved": True}
            print(f"✓ Finished and saved: {namespace}")
        except Exception as e:
            print(f"❌ Error in task {namespace}: {e}")
            traceback.print_exc()
            save_final_results(context, error=str(e))
            context_index[record_key] = {"saved": True}
            continue

    print("\n🎉 ALL TASKS COMPLETED!")


# ==========================================
# 3. 
# ==========================================
def run_pipeline_for_single_record(ctx):
    task_info = ctx["task_info"]
    repo_name = ctx["repo_name"]
    dataset = ctx.get("dataset", DEFAULT_DATASET)
    model_name = ctx.get("model_name", DEFAULT_MODEL)

    socketio.emit('task_start', {'namespace': ctx["namespace"]})
    print(f"Sent 'task_start' for {ctx['namespace']}")

    # ---- STEP 1:  &  ----
    with StageTimer(ctx, "requirement_decomposition"):
        requirement = task_info.get("all_requirement")
        decomposer = RequirementDecomposer(model_name=model_name)
        raw_tasks = decomposer.decompose(requirement)
        ctx["divided_tasks"] = {
            'original_requirement': requirement,
            'decomposed_tasks': [{'id': t.id, 'name': t.name, 'description': t.description} for t in raw_tasks]
        }
        socketio.emit('task_divided', {'tasks': ctx["divided_tasks"]})

    with StageTimer(ctx, "requirement_reflection"):
        reflector = RequirementReflector(model_name=model_name, decomposer=decomposer)
        _, reflection_result = reflector.reflect(requirement, ctx["divided_tasks"]["decomposed_tasks"], socketio)
        ctx["reflection_result"] = reflection_result
        socketio.emit('task_reflection', {'reflection': reflection_result})

    # ---- STEP 2:  UI（） ----
    print(f"Waiting for UI confirmation on 'confirm_tasks'...")
    task_confirmed_event.clear()
    task_confirmed_event.wait()
    approved_tasks = current_ui_feedback["approved_tasks"]['tasks']
    ctx["approved_tasks"] = approved_tasks

    # ---- STEP 3:  &  ----
    with StageTimer(ctx, "library_search"):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {executor.submit(search_and_process_task, t, model_name): t for t in approved_tasks["decomposed_tasks"]}
            task_libraries = [future.result() for future in concurrent.futures.as_completed(futures)]
        task_libraries.sort(key=lambda x: [t['name'] for t in approved_tasks["decomposed_tasks"]].index(x["task_name"]))
        ctx["task_libraries"] = task_libraries
        socketio.emit('task_libraries', {'task_libraries': task_libraries})

    with StageTimer(ctx, "code_retrieval"):
        retriever = CodeRetriever(dataset=dataset)
        dep_analyzer = DependencyAnalyzer(graphs_root=str(get_graphs_dataset_path(dataset)), dataset=dataset)
        repo_data = _load_repo_data(repo_name, dataset)
        task_code_snippets = []
        seen_node_ids = set()
        for task in approved_tasks["decomposed_tasks"]:
            snippets = retriever.retrieve(data=task_info, task_description=task["description"], top_k=1, repo_data=repo_data, seen_node_ids=seen_node_ids)
            upstreams = []
            for snip in snippets:
                dep = dep_analyzer.get_function_calls(
                    task_info["input_code"], repo_name, repo_data, snip["node_id"], 
                    task_info["function_file_path"], task_info["function_name"], task_info["line_num"]
                )
                upstreams.append({"node_id": snip["node_id"], "upstream_calls": dep.callers})
                seen_node_ids.add(snip["node_id"])
            task_code_snippets.append({'task_id': task["name"], 'code_snippets': snippets, 'upstreams': upstreams})
        ctx["task_code_snippets"] = task_code_snippets
        socketio.emit('task_code_snippets', {'task_code_snippets': task_code_snippets})

    # ---- STEP 4:  UI（） ----
    print(f"Waiting for UI confirmation on 'confirm_snippets'...")
    snippets_confirmed_event.clear()
    snippets_confirmed_event.wait()
    ctx["selected_snippets_raw"] = current_ui_feedback["selected_snippets"]
    ctx["selected_libraries"] = current_ui_feedback["selected_libraries"]

    formatted_snippets_list = []
    for i, snippet in enumerate(current_ui_feedback["selected_snippets"], 1):
        raw_path = snippet.get("node_id", "").split("::")[0].replace("\\", "/")
        code_content = snippet.get("code", "")
        class_name = f'class_name: {snippet.get("node_id", "").split("::")[1]}' if snippet.get("node_id", "").split("::")[1] != "" else ""
        formatted_item = f"code{i}:\n# file_path: {raw_path}\n{class_name}\n{code_content}"
        formatted_snippets_list.append(formatted_item)
    ctx["selected_snippets_formatted"] = "\n".join(formatted_snippets_list)

    # ---- STEP 5:  &  ----
    func_root = get_comment_library_path(dataset)
    if dataset == "DevEval":
        project_key = task_info.get("project_path")
    elif dataset == "CoderEvalPython":
        project_key = task_info.get("repo_name")
    else:
        project_key = task_info.get("project_path") or task_info.get("repo_name")

    if not project_key:
        raise KeyError(f"Missing project path key for dataset={dataset}. Expected project_path or repo_name.")

    func_path = os.path.join(func_root, project_key, "functions.jsonl")
    # import_path = os.path.join(func_root, project_key, "import.jsonl")
    # imports = json_to_dict(import_path, "src_file")
    # import_key = project_key + "/" + task_info["function_file_path"].replace("\\", "/")
    # existing_imports = imports.get(import_key, [])
    existing_imports = []
    ctx["existing_imports"] = existing_imports
    in_defs, out_defs = retriever.retrieve_custom_class_definitions_multi(requirement, func_path)
    enhanced_req = f"{requirement}\n"
    if in_defs: enhanced_req += "Input Classes:\n" + "\n".join([c['code'] for c in in_defs]) +"\n"
    if out_defs: enhanced_req += "Return Classes:\n" + "\n".join([c['code'] for c in out_defs])

    generator = CodeGenerator(model_name=model_name)
    with StageTimer(ctx, "first_generation"):
        prompt, generated = generator.generate(
            function_name=task_info["function_name"], input_code=requirement, 
            original_requirement=enhanced_req, related_code=ctx["selected_snippets_formatted"], 
            selected_libraries=current_ui_feedback["selected_libraries"], existing_imports="\n".join(existing_imports), retries=0
        )
        save_first_prompt(ctx, prompt)
        curr_code = generated.code
        socketio.emit('generated_code', {'generated_code': curr_code, 'retries': 0})

    test_agent = get_test_agent(dataset=dataset, model_name=model_name)
    ref_agent = ReflectionAgent(generator, retriever, model_name=model_name)
    if dataset == "CoderEvalPython":
        repo_key = task_info.get("repo_name")
        tests_path = get_comment_library_path(dataset) / repo_key / "tests.jsonl"
        test_codes = load_jsonl(tests_path)
    else:
        test_dir = task_info["repo_dir_path"].split("\\")
        test_codes = load_jsonl(os.path.join(func_root, test_dir[-2], test_dir[-1], "tests.jsonl"))

    retry_count = 0
    while retry_count <= MAX_RETRY:
        print(f"  [Test] Attempt {retry_count} for {ctx['namespace']}")
        t_res = test_agent.test(
            ctx["namespace"],
            curr_code,
            test_codes,
            task_info.get("project_path"),
            sample_id=str(task_info.get("_id", "")),
        )

        ctx["test_history"].append({"retry": retry_count, "passed": t_res.passed, "error": t_res.error_message})
        ctx["generated_history"].append({"retry": retry_count, "code": curr_code})
        socketio.emit('test_result', {'test_result': t_res.error_message, 'test_passed': t_res.passed})

        if t_res.passed:
            print(f"  ✓ Passed!")
            break

        if retry_count < MAX_RETRY:
            retry_count += 1
            stage_name = f"reflection_retry_{retry_count}"
            with StageTimer(ctx, stage_name):
                reflection = ref_agent.reflect_and_regenerate(
                    data=task_info, function_name=repo_name, input_code=task_info["input_code"], class_definitions=enhanced_req,
                    test_result=t_res, current_code=curr_code, retry_count=retry_count, existing_imports=existing_imports, repo_data=repo_data
                )
                curr_code = reflection.code
            socketio.emit('generated_code', {'generated_code': curr_code, 'retries': retry_count})
        else:
            break

    # ---- （， UI） ----
    total_elapsed = sum([v for k, v in ctx["timing"].items() if isinstance(v, (int, float))])
    ctx["timing"]["total_elapsed"] = total_elapsed


# ==========================================
# 4. Socket 
# ==========================================
@socketio.on('confirm_tasks')
def handle_confirm_tasks(data):
    current_ui_feedback["approved_tasks"] = data.get('approved_tasks')
    print(">>> Received confirm_tasks from UI. Resuming pipeline...")
    task_confirmed_event.set()

@socketio.on('confirm_snippets')
def handle_confirm_snippets(data):
    current_ui_feedback["selected_snippets"] = data.get('selected_snippets')
    current_ui_feedback["selected_libraries"] = data.get('selected_libraries')
    print(">>> Received confirm_snippets from UI. Resuming pipeline...")
    snippets_confirmed_event.set()


@socketio.on('clarify_task')
def handle_clarify_task(data):
    decomposer = RequirementDecomposer(model_name=RUN_CONFIG.get("model_name", DEFAULT_MODEL))
    res = decomposer.clarify(data.get("original_requirement"), data["allTasks"], data["name"], data["description"])
    socketio.emit("clarify_result", {"clarify_result": {'id': str(data["index"]), 'name': res.name, 'description': res.description}})

@socketio.on('decompose_further')
def handle_decompose_further(data):
    decomposer = RequirementDecomposer(model_name=RUN_CONFIG.get("model_name", DEFAULT_MODEL))
    res = decomposer.decompose_further(data.get("original_requirement"), data["allTasks"], data["name"], data["description"])
    task_results = [{"id": f"task_{i+int(data['index'])}", "name": t.name, "description": t.description} for i, t in enumerate(res)]
    socketio.emit("decompose_result", {"decompose_result": {"index": str(data["index"]), "subtasks": task_results}})


# ==========================================
# 5. 
# ==========================================
def save_final_results(ctx, error=None):
    result_save_dir = ctx.get("result_save_dir", RESULT_SAVE_ROOT)
    dataset = ctx.get("dataset", DEFAULT_DATASET)
    task_info = ctx.get("task_info", {})
    namespace = ctx.get("namespace", task_info.get("namespace", ""))
    record_key = ctx.get("record_key") or _resolve_record_key(dataset, task_info, namespace)

    output = {
        "record_key": record_key,
        "namespace": namespace,
        "repo_name": ctx.get("repo_name"),
        "dataset": dataset,
        "model_name": ctx.get("model_name"),
        "original_requirement": task_info.get("all_requirement"),
        "initial_divided_tasks": ctx.get("divided_tasks", {}),
        "approved_tasks_for_retrieval": ctx.get("approved_tasks"),
        "reflection_result": ctx.get("reflection_result"),
        "all_recommended_libraries": ctx.get("task_libraries"),
        "selected_libraries": ctx.get("selected_libraries"),
        "all_retrieved_snippets": ctx.get("task_code_snippets"),
        "selected_snippets_for_gen": ctx.get("selected_snippets_formatted"),
        "selected_snippets_raw": ctx.get("selected_snippets_raw"),
        "test_history": ctx.get("test_history", []),
        "generated_history": ctx.get("generated_history", []),
        "timing": ctx.get("timing", {}),
        "error_log": error,
    }

    context_path = _context_all_path(result_save_dir)
    context_index = _load_json_index(context_path)
    context_index[record_key] = output
    _atomic_dump_json(context_path, context_index)


def save_first_prompt(ctx: Dict[str, Any], prompt_text: str):
    result_save_dir = ctx.get("result_save_dir", RESULT_SAVE_ROOT)
    dataset = ctx.get("dataset", DEFAULT_DATASET)
    task_info = ctx.get("task_info", {})
    namespace = ctx.get("namespace", task_info.get("namespace", ""))
    record_key = ctx.get("record_key") or _resolve_record_key(dataset, task_info, namespace)

    prompt_path = _prompt_all_path(result_save_dir)
    prompt_index = _load_json_index(prompt_path)
    prompt_index[record_key] = {
        "record_key": record_key,
        "namespace": namespace,
        "first_gen_prompt": prompt_text,
    }
    _atomic_dump_json(prompt_path, prompt_index)


def search_and_process_task(task, model_name: str):
    try:
        search_agent = SearchAgent(model_name=model_name)
        libs, search_keywords = search_agent.search_libraries(task["description"])
        return {'task_name': task["name"], 'libraries': [{'name': l.name, 'description': l.summary, 'url': l.url} for l in libs], 'keywords': search_keywords}
    except Exception:
        return {'task_name': task["name"], 'libraries': [], 'keywords': []}


def _load_repo_data(repo_name: str, dataset: str) -> Optional[Dict]:
    try:
        pkl_path = get_functions_dataset_path(dataset) / f"{repo_name}.pkl.gz"
        return load_pkl_gz(pkl_path) if pkl_path.exists() else None
    except Exception:
        return None


@socketio.on('connect')
def handle_connect():
    print("Web Client Connected!")


if __name__ == '__main__':
    host = os.getenv("HOST", "0.0.0.0")
    port_str = os.getenv("PORT") or os.getenv("FLASK_RUN_PORT") or "8081"
    try:
        port = int(port_str)
    except ValueError:
        print(f"[WARN] Invalid port {port_str}, fallback to 8081")
        port = 8081

    print(f"[START] CodeAgent server listening on http://{host}:{port}")
    socketio.run(app, host=host, port=port, debug=False)
