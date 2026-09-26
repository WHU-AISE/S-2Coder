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


def _decimal_temperature_grid(start: Decimal, end: Decimal, step: Decimal) -> List[Decimal]:
    values: List[Decimal] = []
    current = start
    guard = 0
    while current <= end + Decimal("0.0000001"):
        values.append(current.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
        current += step
        guard += 1
        if guard > 1000:
            raise RuntimeError("Too many steps when building temperature grid.")
    return values


def _format_temp(value: Decimal) -> str:
    return f"{value.quantize(Decimal('0.1'), rounding=ROUND_HALF_UP):.1f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep gpt-4o-mini temperature from 0.0 to 1.0 (step 0.1), skip 0.7 by default."
    )
    parser.add_argument("--dataset", default="DevEval")
    parser.add_argument("--result-load-dir", default=str(BASE_DIR / "results"))
    parser.add_argument("--max-retry", type=int, default=5)
    parser.add_argument("--max-repo-workers", type=int, default=1)
    parser.add_argument("--start", type=Decimal, default=Decimal("0.0"))
    parser.add_argument("--end", type=Decimal, default=Decimal("1.0"))
    parser.add_argument("--step", type=Decimal, default=Decimal("0.1"))
    parser.add_argument("--skip", default="0.7", help="Comma-separated temperatures to skip, e.g. '0.7,0.9'")
    parser.add_argument(
        "--output-root",
        default=str(BASE_DIR / "results_gpt-4o-mini_temperature_sweep"),
        help="Root directory to store per-temperature ablation outputs",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if not RUN_SCRIPT.exists():
        print(f"[ERROR] Missing script: {RUN_SCRIPT}")
        return 2

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    skip_values = {
        _format_temp(Decimal(item.strip()))
        for item in args.skip.split(",")
        if item.strip()
    }

    temps = _decimal_temperature_grid(args.start, args.end, args.step)
    temps_to_run = [t for t in temps if _format_temp(t) not in skip_values]

    if not temps_to_run:
        print("[WARN] No temperatures left to run after applying --skip.")
        return 0

    print("[Sweep] Planned temperatures:", ", ".join(_format_temp(t) for t in temps_to_run))
    success_count = 0
    failed_count = 0
    skipped_count = 0

    for idx, temp in enumerate(temps_to_run, start=1):
        temp_str = _format_temp(temp)
        save_dir = output_root / f"temp_{temp_str}" / args.dataset
        context_all_path = save_dir / "CONTEXT_ALL.json"
        shard_dir = save_dir / ".tmp_aggregate_shards"
        status_path = output_root / f"temp_{temp_str}.json"

        # If CONTEXT_ALL already exists, treat this temperature as completed and skip it.
        if context_all_path.exists():
            skipped_count += 1
            print(f"\n[Sweep] ({idx}/{len(temps_to_run)}) temperature={temp_str} already completed, skip.")
            print("[Sweep] Found:", context_all_path)
            run_info = {
                "dataset": args.dataset,
                "result_load_dir": args.result_load_dir,
                "model": "gpt-4o-mini",
                "temperature": temp_str,
                "top_p": 1.0,
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
        env["LLM_TEMPERATURE"] = temp_str

        print(f"\n[Sweep] ({idx}/{len(temps_to_run)}) temperature={temp_str} top_p=1.0")
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
            "temperature": temp_str,
            "top_p": 1.0,
            "returncode": rc,
            "elapsed_seconds": elapsed,
            "save_dir": str(save_dir),
            "command": cmd,
        }
        status_path.write_text(json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8")

        if rc != 0:
            failed_count += 1
            print(f"[Sweep] FAILED at temperature={temp_str}, return code={rc}")
        else:
            success_count += 1
            print(f"[Sweep] DONE temperature={temp_str} in {elapsed}s")
    print("\n[Sweep] Completed.")
    print(f"[Sweep] Success: {success_count} / {len(temps_to_run)}")
    print(f"[Sweep] Failed: {failed_count} / {len(temps_to_run)}")
    print(f"[Sweep] Skipped(completed): {skipped_count} / {len(temps_to_run)}")
    print(f"[Sweep] Per-temp status files at: {output_root}")

    return 1 if failed_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
