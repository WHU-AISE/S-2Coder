"""Small, dependency-free file helpers used by AllianceCoder."""

from __future__ import annotations

import gzip
import json
import pickle
from pathlib import Path
from typing import Any, Iterable


def require_path(value: str, option: str) -> Path:
    """Return a configured path or fail with an actionable message."""
    if not value or not value.strip():
        raise ValueError(f"{option} must be configured")
    return Path(value).expanduser()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_pickle_gz(path: Path) -> Any:
    with gzip.open(path, "rb") as handle:
        return pickle.load(handle)
