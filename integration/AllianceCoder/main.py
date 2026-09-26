"""Command-line entry point for the AllianceCoder pipeline."""

from __future__ import annotations

import argparse

from .io_utils import require_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="alliancecoder")
    subparsers = parser.add_subparsers(dest="command", required=True)

    embed = subparsers.add_parser("embed")
    embed.add_argument("--input", default="")
    embed.add_argument("--output", default="")
    embed.add_argument("--model", default="microsoft/unixcoder-base")
    embed.add_argument("--device", default="")

    retrieve = subparsers.add_parser("retrieve")
    retrieve.add_argument("--dataset", required=True)
    retrieve.add_argument("--metadata", default="")
    retrieve.add_argument("--embeddings", default="")
    retrieve.add_argument("--functions-dir", default="")
    retrieve.add_argument("--graph-code-dir", default="")
    retrieve.add_argument("--output", default="")
    retrieve.add_argument("--top-k", type=int, default=1)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "embed":
        from .embeddings import DependencyEmbedder

        embedder = DependencyEmbedder(args.model, args.device or None)
        embedder.process_file(
            require_path(args.input, "--input"),
            require_path(args.output, "--output"),
        )
        return

    from .retrieval import DependencyRetriever

    graph_code_dir = (
        require_path(args.graph_code_dir, "--graph-code-dir")
        if args.graph_code_dir
        else None
    )
    retriever = DependencyRetriever(
        require_path(args.functions_dir, "--functions-dir"),
        graph_code_dir,
        args.top_k,
    )
    retriever.process(
        args.dataset,
        require_path(args.metadata, "--metadata"),
        require_path(args.embeddings, "--embeddings"),
        require_path(args.output, "--output"),
    )


if __name__ == "__main__":
    main()
