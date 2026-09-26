#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a 10-case questionnaire package from filtered DevEval cases."
    )
    parser.add_argument(
        "--intersection-with-code",
        default="",
    )
    parser.add_argument(
        "--dataset-info",
        default="",
    )
    parser.add_argument(
        "--context-all",
        default="",
    )
    parser.add_argument(
        "--num-cases",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--output-json",
        default="",
    )
    parser.add_argument(
        "--output-md",
        default="",
    )
    return parser.parse_args()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def build_dataset_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        ns = row.get("namespace")
        if ns and ns not in index:
            index[ns] = row
    return index


def pick_cases(items: List[Dict[str, Any]], k: int, seed: int) -> List[Dict[str, Any]]:
    if k >= len(items):
        return items
    rnd = random.Random(seed)
    idxs = list(range(len(items)))
    rnd.shuffle(idxs)
    selected = [items[i] for i in idxs[:k]]
    selected.sort(key=lambda x: x.get("namespace", ""))
    return selected


def build_stage_info(context_item: Dict[str, Any]) -> Dict[str, Any]:
    # Keep stage fields explicit for easier human evaluation.
    keys = [
        "original_requirement",
        "enhanced_requirement",
        "initial_divided_tasks",
        "approved_tasks_for_retrieval",
        "reflection_result",
        "all_recommended_libraries",
        "selected_libraries",
        "all_retrieved_snippets",
        "selected_snippets_for_gen",
        "selected_snippets_raw",
        "generated_history",
        "test_history",
        "timing",
        "error_log",
    ]
    return {k: context_item.get(k) for k in keys if k in context_item}


def to_markdown(questionnaire: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# DevEval 10-Case Questionnaire")
    lines.append("")
    lines.append(f"- Total cases: {questionnaire['num_cases']}")
    lines.append(f"- Sampling seed: {questionnaire['seed']}")
    lines.append("")

    for i, case in enumerate(questionnaire["cases"], start=1):
        lines.append(f"## Case {i}: `{case['namespace']}`")
        lines.append("")
        lines.append("### Original Requirement")
        lines.append(case["original_requirement"] or "")
        lines.append("")

        lines.append("### Scoring (0-4)")
        lines.append("- Correctness: ____")
        lines.append("- Code Quality: ____")
        lines.append("- Maintainability: ____")
        lines.append("")

        lines.append("### Generated Results")
        for method, code in case["generated_results"].items():
            lines.append(f"#### {method}")
            lines.append("```python")
            lines.append(code or "")
            lines.append("```")
        lines.append("")

        lines.append("### CONTEXT_ALL Stage Info")
        lines.append("```json")
        lines.append(json.dumps(case["context_all_stage_info"], ensure_ascii=False, indent=2))
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()

    inter = load_json(Path(args.intersection_with_code))
    dataset_rows = load_json(Path(args.dataset_info))
    context_all = load_json(Path(args.context_all))

    items = inter.get("items", [])
    if not isinstance(dataset_rows, list):
        raise ValueError("dataset-info must be a JSON list")
    if not isinstance(context_all, dict):
        raise ValueError("context-all must be a dict keyed by namespace")

    selected_items = pick_cases(items, args.num_cases, args.seed)
    dataset_by_ns = build_dataset_index(dataset_rows)

    cases = []
    for item in selected_items:
        ns = item.get("namespace")
        ds = dataset_by_ns.get(ns, {})
        ctx = context_all.get(ns, {})

        original_requirement = (
            ds.get("all_requirement")
            or ds.get("human_requeriment")
            or ctx.get("original_requirement")
            or ""
        )

        case = {
            "namespace": ns,
            "repo_name": ds.get("repo_name") or ctx.get("repo_name"),
            "project_path": ds.get("project_path") or ctx.get("project_path"),
            "original_requirement": original_requirement,
            "generated_results": item.get("codes", {}),
            "context_all_stage_info": build_stage_info(ctx),
            "scores": {
                "Correctness": {"score": None, "min": 0, "max": 4, "comment": ""},
                "Code Quality": {"score": None, "min": 0, "max": 4, "comment": ""},
                "Maintainability": {"score": None, "min": 0, "max": 4, "comment": ""},
            },
        }
        cases.append(case)

    result = {
        "task": "DevEval questionnaire",
        "num_cases": len(cases),
        "seed": args.seed,
        "sources": {
            "intersection_with_code": args.intersection_with_code,
            "dataset_info": args.dataset_info,
            "context_all": args.context_all,
        },
        "cases": cases,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    out_md = Path(args.output_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(to_markdown(result), encoding="utf-8")

    print("[Done]")
    print(f"cases: {len(cases)}")
    print(f"output_json: {out_json}")
    print(f"output_md: {out_md}")


if __name__ == "__main__":
    main()
