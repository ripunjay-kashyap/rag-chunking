"""CLI entry point: python -m rag_chunking <command>."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from rag_chunking.chunking import make_chunker
from rag_chunking.config import DEFAULT_CONFIG, Config, ConfigError, load_config
from rag_chunking.inspection import cut_summary, find_cuts, format_table, summary_stats
from rag_chunking.normalize import normalize
from rag_chunking.results import sha256, write_json


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
    document = normalize(config.settings.document.read_text(encoding="utf-8"))
    chunks = make_chunker(run.params).chunk(document)
    stats = summary_stats(chunks, document)
    cuts = find_cuts(chunks, document)
    out = config.settings.results_dir / run.id / "chunks.json"
    write_json(
        out,
        {
            "run_id": run.id,
            "strategy": run.strategy,
            "params": asdict(run.params),
            "document_sha256": sha256(document),
            "stats": stats,
            "cut_summary": cut_summary(cuts),
            "cuts": [asdict(c) for c in cuts],
            "chunks": [c.to_dict() for c in chunks],
        },
    )
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
            if args.run_id:
                config.run(args.run_id)
            return _not_implemented("run", "P6–P10")
        if args.command == "review":
            config.run(args.run)
            return _not_implemented("review", "P9")
        if args.command == "report":
            return _not_implemented("report", "P12")
    except ConfigError as exc:
        print(f"Config error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
