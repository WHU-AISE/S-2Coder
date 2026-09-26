#!/usr/bin/env python3
"""
Build a mapping between Parser.node_map keys (GraphML/plain keys) and CoderEvalJava record _id.

User rule (as requested):
- For each node_map key, split by "/"
- parts[0] should match record.file_path with "/" replaced by "\\"
- the token right BEFORE the first numeric token matches record.name
- the first numeric token matches record.lineno
- if matched, map node_key -> record._id

Usage example:
  conda run -n reposcope python scripts/map_node_keys_to_coder_eval_ids.py \
    --parser-pkl /home/inspur/lyl/RepoScope/.cache/CoderEvalJava/atmosphere/code_graph/bge-small_69f15cae_parser.pkl \
    --dataset-json /home/inspur/lyl/RepoScope/datasets/CoderEvalJava.json \
    --out /home/inspur/lyl/RepoScope/.cache/CoderEvalJava/atmosphere/code_graph/nodekey_to_id.json
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import re
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Tuple


_NUM_RE = re.compile(r"^\d+$")

# Ensure RepoScope root (which contains `src/`) is importable so pickle.load works
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _first_numeric_index(parts: List[str]) -> Optional[int]:
    for i, p in enumerate(parts):
        if _NUM_RE.match(p):
            return i
    return None


def _load_coder_eval_records(dataset_json_path: str) -> List[dict]:
    with open(dataset_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    records = obj.get("RECORDS")
    if not isinstance(records, list):
        raise ValueError(f"Expected top-level key 'RECORDS' to be a list in {dataset_json_path}")
    return records


def _index_records(records: List[dict]) -> Dict[str, List[dict]]:
    """
    Index by file_path with '/' replaced by '\\' (exactly as requested).
    """
    by_file: Dict[str, List[dict]] = defaultdict(list)
    for rec in records:
        fp = rec.get("file_path")
        if not isinstance(fp, str):
            continue
        by_file[fp.replace("/", "\\")].append(rec)
    return by_file


def _match_node_key_to_id(
    node_key: str,
    records_by_file: Dict[str, List[dict]],
    max_line_delta: int = 0,
) -> Optional[str]:
    parts = node_key.split("/")
    if not parts:
        return None
    file_part = parts[0]
    idx = _first_numeric_index(parts)
    if idx is None or idx == 0:
        return None
    name_part = parts[idx - 1]
    try:
        lineno_part = int(parts[idx])
    except ValueError:
        return None

    candidates = records_by_file.get(file_part)
    if not candidates:
        return None

    for rec in candidates:
        if rec.get("name") != name_part:
            continue
        try:
            rec_lineno = int(rec.get("lineno"))
        except Exception:
            continue
        if rec_lineno != lineno_part:
            continue
        _id = rec.get("_id")
        if isinstance(_id, str) and _id:
            return _id
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parser-pkl", required=True, help="Path to *_parser.pkl")
    ap.add_argument("--dataset-json", required=True, help="Path to datasets/CoderEvalJava.json")
    ap.add_argument("--out", required=True, help="Output JSON path for node_key -> _id mapping")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit for number of node_map keys to process (0=all)")
    ap.add_argument("--debug-id", default="", help="Optional CoderEvalJava record _id to debug matching against node_map keys")
    # Important: unpickling imports RepoScope modules, which trigger Config.init() that parses CLI args.
    # We provide a dedicated config name and temporarily rewrite sys.argv during pickle.load.
    ap.add_argument(
        "--reposcope-config",
        default=os.environ.get("CODE_GRAPH_CONFIG_FILE", "config_rq1"),
        help="RepoScope config yaml name (without .yaml) used during unpickling (default: env CODE_GRAPH_CONFIG_FILE or config_rq1)",
    )
    args = ap.parse_args()

    # Make RepoScope's Config.init() happy during unpickling: it expects only '--config <name>'.
    orig_argv = sys.argv[:]
    try:
        os.environ["CODE_GRAPH_CONFIG_FILE"] = args.reposcope_config
        sys.argv = [orig_argv[0], "--config", args.reposcope_config]
        with open(args.parser_pkl, "rb") as f:
            parser_obj = pickle.load(f)
    finally:
        sys.argv = orig_argv

    node_map = getattr(parser_obj, "node_map", None)
    if not isinstance(node_map, dict):
        raise ValueError("Loaded parser.pkl does not contain a dict attribute 'node_map'")

    records = _load_coder_eval_records(args.dataset_json)
    by_file = _index_records(records)
    by_id = {r.get("_id"): r for r in records if isinstance(r.get("_id"), str)}

    keys = list(node_map.keys())
    if args.limit and args.limit > 0:
        keys = keys[: args.limit]

    mapping: Dict[str, str] = {}
    missing_file = 0
    missing_numeric = 0
    miss = 0

    for k in keys:
        if not isinstance(k, str):
            continue
        parts = k.split("/")
        if not parts:
            miss += 1
            continue
        if parts[0] not in by_file:
            missing_file += 1
        idx = _first_numeric_index(parts)
        if idx is None:
            missing_numeric += 1
        _id = _match_node_key_to_id(k, by_file, max_line_delta=args.max_line_delta)
        if _id is None:
            miss += 1
            continue
        mapping[k] = _id

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(mapping, f, ensure_ascii=False, indent=2)

    total = len(keys)
    hit = len(mapping)
    print(f"node_map keys processed: {total}")
    print(f"matched (node_key -> _id): {hit}")
    print(f"match rate: {hit / max(1, total):.4f}")
    print(f"missing file_part in dataset index: {missing_file}")
    print(f"missing numeric segment in key: {missing_numeric}")
    if hit:
        # show a few samples
        for i, (k, v) in enumerate(list(mapping.items())[:5]):
            print(f"sample[{i}]: {v}  <=  {k}")

    if args.debug_id:
        rec = by_id.get(args.debug_id)
        if not rec:
            print(f"[debug] _id not found in dataset: {args.debug_id}")
            return
        fp = rec.get("file_path")
        name = rec.get("name")
        lineno = rec.get("lineno")
        fp_key = fp.replace("/", "\\") if isinstance(fp, str) else ""
        try:
            lineno_i = int(lineno)
        except Exception:
            lineno_i = None
        print("\n[debug] record:")
        print("  _id:", rec.get("_id"))
        print("  file_part:", fp_key)
        print("  name:", name)
        print("  lineno:", lineno)

        same_file = [k for k in keys if isinstance(k, str) and k.split("/")[0] == fp_key]
        print("[debug] node_keys with same file_part:", len(same_file))
        if not same_file:
            # help spot path normalization issues
            close = [k for k in keys if isinstance(k, str) and fp_key.split("\\")[-1] in k][:10]
            print("[debug] no exact file_part match; showing up to 10 keys that contain basename:")
            for k in close:
                print("   ", k)
            return

        # derive (name_token, first_lineno) for same-file keys
        derived = []
        for k in same_file:
            parts = k.split("/")
            idx = _first_numeric_index(parts)
            if idx is None or idx == 0:
                continue
            nm = parts[idx - 1]
            try:
                ln = int(parts[idx])
            except Exception:
                continue
            derived.append((nm, ln, k))

        same_name = [t for t in derived if t[0] == name]
        print("[debug] same file_part & name_token matches:", len(same_name))
        if lineno_i is not None and same_name:
            lns = sorted({t[1] for t in same_name})
            print("[debug] candidate first line numbers for this name (min/max/unique):", min(lns), max(lns), len(lns))
            closest = sorted(lns, key=lambda x: abs(x - lineno_i))[:10]
            print("[debug] closest to dataset lineno:", closest)
            print("[debug] example keys (up to 5):")
            for _, _, k in same_name[:5]:
                print("   ", k)
        else:
            # show most common derived name tokens in this file
            from collections import Counter
            c = Counter([t[0] for t in derived])
            print("[debug] top derived name tokens in this file:", c.most_common(10))


if __name__ == "__main__":
    main()


