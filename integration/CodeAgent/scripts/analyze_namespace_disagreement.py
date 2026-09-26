#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Find namespaces that are correct in CONTEXT_ALL.json but wrong in all other result files."
        )
    )
    parser.add_argument(
        "--context",
        default="",
        help="Path to CONTEXT_ALL.json",
    )
    parser.add_argument(
        "--wrong-files",
        nargs="+",
        default=[
            "",
            "",
            "",
            "",
            "",
            "",
        ],
        help="Result files expected to be wrong (indexed by namespace)",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--output-txt",
        default="",
        help="Output TXT file path (one namespace per line)",
    )
    parser.add_argument(
        "--assume-no-status-error",
        action="store_true",
        default=True,
        help=(
            "Treat records without explicit status field as error. "
            "Enabled by default because some files (e.g. gpt-4o-mini_devEval172.jsonl) have no status field."
        ),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{i} JSON decode error: {exc}") from exc
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def normalize_status(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    if isinstance(v, int):
        if v == 1:
            return True
        if v == 0:
            return False
        return None
    if isinstance(v, str):
        s = v.strip().lower()
        if s in {"pass", "passed", "success", "true", "1", "ok"}:
            return True
        if s in {"error", "fail", "failed", "false", "0", "wrong"}:
            return False
    return None


def get_context_passed(entry: Dict[str, Any]) -> Optional[bool]:
    th = entry.get("test_history")
    if isinstance(th, list) and th:
        last = th[-1]
        if isinstance(last, dict) and "passed" in last:
            return normalize_status(last.get("passed"))

    # fallback fields if needed
    for key in ("Result", "result", "passed", "is_solved"):
        if key in entry:
            return normalize_status(entry.get(key))
    return None


def get_record_status(record: Dict[str, Any]) -> Optional[bool]:
    # explicit status fields first
    for key in ("Result", "result", "is_solved", "passed"):
        if key in record:
            st = normalize_status(record.get(key))
            if st is not None:
                return st

    # CodeAgent-style fallback: use python_repl tool result in tool_logs
    tool_logs = record.get("tool_logs")
    if isinstance(tool_logs, list):
        repl_flags: List[bool] = []
        for log in tool_logs:
            if not isinstance(log, dict):
                continue
            if log.get("tool") != "python_repl":
                continue
            ok = log.get("ok")
            if isinstance(ok, bool):
                repl_flags.append(ok)
        if repl_flags:
            # If any python_repl run failed, treat as failure.
            # Otherwise if all are True, treat as pass.
            return all(repl_flags)

    # fallback from test_history
    th = record.get("test_history")
    if isinstance(th, list) and th:
        last = th[-1]
        if isinstance(last, dict) and "passed" in last:
            return normalize_status(last.get("passed"))

    return None


def iter_records(path: Path) -> Iterable[Dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        yield from read_jsonl(path)
        return

    data = read_json(path)
    if isinstance(data, dict):
        # CONTEXT_ALL-like dict(namespace -> entry)
        for k, v in data.items():
            if isinstance(v, dict):
                if "namespace" not in v:
                    v = dict(v)
                    v["namespace"] = k
                yield v
    elif isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                yield item


def collect_context_correct_namespaces(path: Path) -> Set[str]:
    data = read_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict in context file: {path}")

    correct: Set[str] = set()
    for ns, entry in data.items():
        if not isinstance(entry, dict):
            continue
        passed = get_context_passed(entry)
        if passed is True:
            correct.add(entry.get("namespace") or ns)
    return correct


def collect_wrong_namespaces(
    path: Path, assume_no_status_error: bool = True
) -> Tuple[Set[str], int, int, int]:
    wrong: Set[str] = set()
    total = 0
    explicit_status = 0
    missing_status = 0

    for rec in iter_records(path):
        ns = rec.get("namespace")
        if not ns:
            continue
        total += 1
        st = get_record_status(rec)
        if st is None:
            missing_status += 1
            if assume_no_status_error:
                wrong.add(ns)
            continue

        explicit_status += 1
        if st is False:
            wrong.add(ns)

    return wrong, total, explicit_status, missing_status


def main() -> None:
    args = parse_args()

    context_path = Path(args.context)
    if not context_path.exists():
        raise FileNotFoundError(f"Context file not found: {context_path}")

    wrong_files = [Path(p) for p in args.wrong_files]
    unique_wrong_files: List[Path] = []
    seen = set()
    for p in wrong_files:
        rp = str(p.resolve())
        if rp in seen:
            continue
        seen.add(rp)
        unique_wrong_files.append(p)

    for p in unique_wrong_files:
        if not p.exists():
            raise FileNotFoundError(f"Wrong file not found: {p}")

    context_correct = collect_context_correct_namespaces(context_path)

    wrong_sets: List[Set[str]] = []
    stats: List[Dict[str, Any]] = []
    for p in unique_wrong_files:
        wrong_ns, total, explicit_status, missing_status = collect_wrong_namespaces(
            p, assume_no_status_error=args.assume_no_status_error
        )
        wrong_sets.append(wrong_ns)
        stats.append(
            {
                "file": str(p),
                "total_records_with_namespace": total,
                "explicit_status_records": explicit_status,
                "missing_status_records": missing_status,
                "wrong_namespace_count": len(wrong_ns),
            }
        )

    if wrong_sets:
        wrong_intersection = set.intersection(*wrong_sets)
    else:
        wrong_intersection = set()

    target = sorted(context_correct & wrong_intersection)

    output_obj = {
        "context_file": str(context_path),
        "wrong_files": [str(p) for p in unique_wrong_files],
        "assume_no_status_error": args.assume_no_status_error,
        "context_correct_count": len(context_correct),
        "wrong_intersection_count": len(wrong_intersection),
        "final_count": len(target),
        "namespaces": target,
        "file_stats": stats,
    }

    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(output_obj, ensure_ascii=False, indent=2), encoding="utf-8")

    out_txt = Path(args.output_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(target) + ("\n" if target else ""), encoding="utf-8")

    print("[Done]")
    print(f"context_correct_count: {len(context_correct)}")
    print(f"wrong_intersection_count: {len(wrong_intersection)}")
    print(f"final_count: {len(target)}")
    print(f"output_json: {out_json}")
    print(f"output_txt: {out_txt}")
    print("\n[Per-file stats]")
    for s in stats:
        print(
            f"- {s['file']}\n"
            f"  total={s['total_records_with_namespace']}, "
            f"explicit_status={s['explicit_status_records']}, "
            f"missing_status={s['missing_status_records']}, "
            f"wrong_ns={s['wrong_namespace_count']}"
        )


if __name__ == "__main__":
    main()
