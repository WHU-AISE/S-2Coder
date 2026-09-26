"""Retrieve repository code for embedded dependency descriptions."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .io_utils import load_json, load_jsonl, load_pickle_gz, write_json


def cosine_similarity(left: Iterable[float], right: Iterable[float]) -> float:
    left_vector = np.asarray(list(left), dtype=float)
    right_vector = np.asarray(list(right), dtype=float)
    if left_vector.shape != right_vector.shape:
        return 0.0
    denominator = np.linalg.norm(left_vector) * np.linalg.norm(right_vector)
    if denominator == 0:
        return 0.0
    return float(np.dot(left_vector, right_vector) / denominator)


class DependencyRetriever:
    """Rank repository functions for each dependency embedding."""

    def __init__(
        self,
        functions_dir: Path,
        graph_code_dir: Path | None = None,
        top_k: int = 1,
    ) -> None:
        self.functions_dir = functions_dir
        self.graph_code_dir = graph_code_dir
        self.top_k = top_k

    @staticmethod
    def _is_excluded(node_id: str, task: dict[str, Any], dataset: str) -> bool:
        parts = node_id.split("::")
        if len(parts) < 3:
            return True
        same_target = (
            parts[0] == task.get("function_file_path")
            and parts[2] == task.get("function_name")
        )
        if same_target:
            return True
        if dataset != "DevEval" and (
            "passk_validte" in node_id or "\\test_" in node_id
        ):
            return True
        return "build\\lib" in node_id or "site-packages" in node_id

    def _load_code_index(self, dataset: str, repo_name: str) -> dict[str, str]:
        if self.graph_code_dir is None:
            return {}
        path = self.graph_code_dir / dataset / f"{repo_name}_all.json"
        if not path.exists():
            return {}
        value = load_json(path)
        return value if isinstance(value, dict) else {}

    def retrieve_task(
        self,
        dataset: str,
        task: dict[str, Any],
        dependency_record: dict[str, Any],
    ) -> dict[str, Any]:
        repo_name = str(task["repo_name"])
        function_data = load_pickle_gz(
            self.functions_dir / dataset / f"{repo_name}.pkl.gz"
        )
        code_index = self._load_code_index(dataset, repo_name)
        all_results: list[list[dict[str, Any]]] = []

        for dependency_embedding in dependency_record.get("emb", []):
            ranked: list[tuple[float, str, str]] = []
            for node_id, node_info in function_data.items():
                if self._is_excluded(node_id, task, dataset):
                    continue
                annotation_embedding = node_info.get("annotation_embedding")
                if annotation_embedding is None:
                    continue
                score = cosine_similarity(
                    dependency_embedding, annotation_embedding
                )
                code = node_info.get("code") or code_index.get(node_id, "")
                ranked.append((score, node_id, code))

            ranked.sort(key=lambda item: item[0], reverse=True)
            all_results.append(
                [
                    {"node_id": node_id, "score": score, "code": code}
                    for score, node_id, code in ranked[: self.top_k]
                ]
            )

        return {
            "namespace": task.get("namespace", ""),
            "dependency_matches": all_results,
        }

    def process(
        self,
        dataset: str,
        metadata_path: Path,
        embeddings_path: Path,
        output_path: Path,
    ) -> None:
        metadata = load_json(metadata_path)
        if isinstance(metadata, dict):
            metadata = metadata.get("RECORDS", [])
        dependencies = load_jsonl(embeddings_path)
        if len(metadata) != len(dependencies):
            raise ValueError(
                "Dataset metadata and dependency embeddings have different lengths"
            )
        results = [
            self.retrieve_task(dataset, task, dependency)
            for task, dependency in zip(metadata, dependencies)
        ]
        write_json(output_path, results)
