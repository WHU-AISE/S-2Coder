"""Create embeddings for dependency descriptions."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch

from .io_utils import load_jsonl, write_jsonl
from .unixcoder import UniXcoder


DEPENDENCY_PATTERN = re.compile(
    r"(?:\*\*)?`[^`]+`(?:\*\*)?:\s(.*?)(?=\n\d+\.|$)",
    re.DOTALL,
)


class DependencyEmbedder:
    """Extract dependency descriptions and encode them with UniXcoder."""

    def __init__(
        self,
        model_name: str = "microsoft/unixcoder-base",
        device: str | None = None,
    ) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = UniXcoder(model_name)
        self.model.to(self.device)

    def encode(self, text: str) -> list[float]:
        token_ids = self.model.tokenize(
            [text], max_length=512, mode="<encoder-only>"
        )
        source_ids = torch.tensor(token_ids).to(self.device)
        with torch.no_grad():
            _, text_embedding = self.model(source_ids)
        normalized = torch.nn.functional.normalize(text_embedding, p=2, dim=1)
        return normalized.detach().cpu().numpy().flatten().tolist()

    def process_record(self, record: dict[str, Any]) -> dict[str, Any]:
        raw_dependencies = str(record.get("all_dependencies", ""))
        descriptions = [
            match.strip()
            for match in DEPENDENCY_PATTERN.findall(raw_dependencies)
            if match.strip()
        ]
        if not descriptions:
            record_id = record.get("_id", record.get("namespace", "unknown"))
            raise ValueError(f"No dependency descriptions found for {record_id}")

        result = dict(record)
        result["all_dependencies"] = descriptions
        result["emb"] = [self.encode(description) for description in descriptions]
        return result

    def process_file(self, input_path: Path, output_path: Path) -> None:
        records = load_jsonl(input_path)
        write_jsonl(output_path, (self.process_record(record) for record in records))
