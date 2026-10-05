"""CLI entry point: python -m rag_chunking <command>."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag_chunking import pipeline
from rag_chunking.config import DEFAULT_CONFIG, Config, ConfigError, load_config
from rag_chunking.embeddings import make_embedder
from rag_chunking.evaluation.answer_key import AnswerKeyError
from rag_chunking.inspection import cut_summary, format_table


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag_chunking",
        description="Compare fixed-size and structure-aware chunking for RAG.",
    )
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help=f"default: {DEFAULT_CONFIG}"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    chunk = sub.add_parser("chunk", help="chunk the document and inspect chunks (no API calls)")
    chunk.add_argument("--run", required=True, help="run id from the config")

    run = sub.add_parser("run", help="chunk, embed, retrieve, generate and auto-check")
    target = run.add_mutually_exclusive_group(required=True)
    target.add_argument("--run", dest="run_id", help="run id from the config")
    target.add_argument("--all", action="store_true", help="every run in the config")

    review = sub.add_parser("review", help="create or refresh results/<run>/review.csv")
    review.add_argument("--run", required=True, help="run id from the config")

    sub.add_parser("report", help="build results/comparison.md from all runs")
    sub.add_parser("list", help="list the runs defined in the config")
    return parser


def _not_implemented(command: str, phase: str) -> int:
    print(f"'{command}' is not implemented yet ({phase}).", file=sys.stderr)
    return 2


def _cmd_chunk(config: Config, run_id: str) -> int:
    run = config.run(run_id)
    chunked = pipeline.chunk_run(config, run, pipeline.load_document(config))
    chunks, stats, cuts = chunked.chunks, chunked.stats, chunked.cuts
    out = config.settings.results_dir / run.id / "chunks.json"
    print(format_table(chunks))
    print()
    print("stats:", ", ".join(f"{k}={v}" for k, v in stats.items()))
    summary = ", ".join(f"{k}={v}" for k, v in cut_summary(cuts).items())
    print(f"boundaries ({len(cuts)} chunk starts + ends): {summary}")
    for cut in cuts:
        if cut.side == "end" and (cut.mid_word or cut.kind not in ("clean", "sentence-end")):
            flag = "mid-word" if cut.mid_word else ""
            print(f"  {cut.chunk_id:<12} {cut.kind:<13} {flag:<9} {cut.context}")
    print(f"\nwrote {out}")
    return 0


def _cmd_run(config: Config, run_ids: list[str]) -> int:
    # One embedder for all runs, so runs that share chunks (A-500 / A-500-budget) share
    # the cache within a single invocation too.
    embedder = make_embedder(config.embedding, config.settings.cache_dir)
    for run_id in run_ids:
        m = pipeline.execute(config, config.run(run_id), embedder)
        cache = m["embedding_cache"]
        print(
            f"{run_id:<16} chunks={m['chunk_stats']['chunks']:<3} "
            f"embedding calls={m['api_calls']['embedding']} "
            f"(cache hits={cache['hits']}, misses={cache['misses']}) "
            f"mean context tokens={m['mean_context_tokens']}"
        )
    print("generation is not implemented yet (P8).")
    return 0


def _cmd_list(config: Config) -> int:
    for run in config.runs.values():
        print(f"{run.id:<18} {run.strategy:<10} {run.retrieval:<13} {run.params}")
    return 0


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
    args = _build_parser().parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "list":
            return _cmd_list(config)
        if args.command == "chunk":
            return _cmd_chunk(config, args.run)
        if args.command == "run":
            run_ids = list(config.runs) if args.all else [config.run(args.run_id).id]
            return _cmd_run(config, run_ids)
        if args.command == "review":
            config.run(args.run)
            return _not_implemented("review", "P9")
        if args.command == "report":
            return _not_implemented("report", "P12")
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1
    except AnswerKeyError as exc:
        print(f"Answer key error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
