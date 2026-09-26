import argparse
import json
import os
import subprocess
import sys
import time
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import List


BASE_DIR = Path(__file__).resolve().parent
RUN_SCRIPT = BASE_DIR / "run_ablation_study.py"


def _format_top_p(value: Decimal) -> str:
    normalized = value.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    s = format(normalized, "f")
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if "." not in s:
        s = f"{s}.0"
    return s


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep gpt-4o-mini top_p with temperature fixed at 0.0."
    )
    parser.add_argument("--dataset", default="DevEval")
    parser.add_argument("--result-load-dir", default=str(BASE_DIR / "results"))
    parser.add_argument("--max-retry", type=int, default=5)
    parser.add_argument("--max-repo-workers", type=int, default=1)
    parser.add_argument(
        "--top-p-values",
        default="1.0,0.95,0.9,0.8,0.7,0.6,0.5,0.3,0.1",
        help="Comma-separated top_p values, e.g. '1.0,0.95,0.9'",
    )
    parser.add_argument(
        "--output-root",
        default=str(BASE_DIR / "results_gpt-4o-mini_top_p_sweep"),
        help="Root directory to store per-top_p ablation outputs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def _parse_top_p_values(values: str) -> List[Decimal]:
    parsed: List[Decimal] = []
    for item in values.split(","):
        s = item.strip()
        if not s:
            continue
        try:
            v = Decimal(s)
        except Exception as exc:
            raise ValueError(f"Invalid top_p value: {s}") from exc
        if v < Decimal("0") or v > Decimal("1"):
            raise ValueError(f"top_p out of range [0,1]: {s}")
        parsed.append(v)
    if not parsed:
        raise ValueError("No valid top_p values found.")
    return parsed


def main() -> int:
    args = parse_args()

    if not RUN_SCRIPT.exists():
        print(f"[ERROR] Missing script: {RUN_SCRIPT}")
        return 2

    try:
        top_p_values = _parse_top_p_values(args.top_p_values)
    except ValueError as err:
        print(f"[ERROR] {err}")
        return 2

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    print("[Sweep] Planned top_p values:", ", ".join(_format_top_p(v) for v in top_p_values))
    print("[Sweep] Fixed temperature: 0.0")

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for idx, top_p in enumerate(top_p_values, start=1):
        top_p_str = _format_top_p(top_p)
        save_dir = output_root / f"top_p_{top_p_str}" / args.dataset
        context_all_path = save_dir / "CONTEXT_ALL.json"
        shard_dir = save_dir / ".tmp_aggregate_shards"
        status_path = output_root / f"top_p_{top_p_str}.json"

        if context_all_path.exists():
            skipped_count += 1
            print(f"\n[Sweep] ({idx}/{len(top_p_values)}) top_p={top_p_str} already completed, skip.")
            print("[Sweep] Found:", context_all_path)
            run_info = {
                "dataset": args.dataset,
                "result_load_dir": args.result_load_dir,
                "model": "gpt-4o-mini",
                "temperature": "0.0",
                "top_p": top_p_str,
                "status": "skipped_completed",
                "returncode": 0,
                "elapsed_seconds": 0.0,
                "save_dir": str(save_dir),
                "command": [],
            }
            status_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")
            continue

        save_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(RUN_SCRIPT),
            "--dataset",
            args.dataset,
            "--result-load-dir",
            args.result_load_dir,
            "--max-retry",
            str(args.max_retry),
            "--max-repo-workers",
            str(args.max_repo_workers),
            "--generation-model",
            "gpt-4o-mini",
            "--general-model",
            "gpt-4o-mini",
            "--ablation-save-dir",
            str(save_dir),
        ]

        env = os.environ.copy()
        env["LLM_PROVIDER"] = "openai"
        env["LLM_MODEL"] = "gpt-4o-mini"
        env["LLM_TEMPERATURE"] = "0.0"
        env["LLM_TOP_P"] = top_p_str

        print(f"\n[Sweep] ({idx}/{len(top_p_values)}) temperature=0.0 top_p={top_p_str}")
        print("[Sweep] Output dir:", save_dir)
        if shard_dir.exists():
            shard_files = sorted(shard_dir.glob("context_*.json"))
            print(
                f"[Sweep] Resume mode detected: {shard_dir} exists "
                f"({len(shard_files)} shard files). Already-generated tasks will be skipped."
            )
        print("[Sweep] Command:", " ".join(cmd))

        start_ts = time.time()
        if args.dry_run:
            rc = 0
        else:
            rc = subprocess.run(cmd, cwd=str(BASE_DIR), env=env).returncode
        elapsed = round(time.time() - start_ts, 2)

        run_info = {
            "dataset": args.dataset,
            "result_load_dir": args.result_load_dir,
            "model": "gpt-4o-mini",
            "temperature": "0.0",
            "top_p": top_p_str,
            "returncode": rc,
            "elapsed_seconds": elapsed,
            "save_dir": str(save_dir),
            "command": cmd,
        }
        status_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

        if rc != 0:
            failed_count += 1
            print(f"[Sweep] FAILED at top_p={top_p_str}, return code={rc}")
        else:
            success_count += 1
            print(f"[Sweep] DONE top_p={top_p_str} in {elapsed}s")

    print("\n[Sweep] Completed.")
    print(f"[Sweep] Success: {success_count} / {len(top_p_values)}")
    print(f"[Sweep] Failed: {failed_count} / {len(top_p_values)}")
    print(f"[Sweep] Skipped(completed): {skipped_count} / {len(top_p_values)}")
    print(f"[Sweep] Per-top_p status files at: {output_root}")

    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
