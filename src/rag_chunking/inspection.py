"""Chunk statistics and boundary classification, for eyeballing chunks before any API call."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any

from rag_chunking.chunking.base import Chunk
from rag_chunking.tokens import estimate_tokens

# Where a boundary falls, worst first. Structure wins over prose: a cut through a table
# row or list item is the failure the assignment asks about. "clean" = on a line break.
CUT_KINDS = (
    "table-row",
    "list-item",
    "heading",
    "mid-sentence",
    "sentence-end",
    "clean",
)


@dataclass(frozen=True)
class Cut:
    chunk_id: str
    side: str  # "start" | "end"
    position: int
    kind: str
    mid_word: bool
    context: str  # a few characters either side, with "|" marking the cut


def summary_stats(chunks: list[Chunk], document: str) -> dict[str, Any]:
    tokens = [c.est_tokens for c in chunks]
    total = sum(tokens)
    return {
        "chunks": len(chunks),
        "mean_tokens": round(statistics.mean(tokens), 1),
        "min_tokens": min(tokens),
        "max_tokens": max(tokens),
        "total_tokens": total,
        # > 1.0 means text is stored more than once (overlap, heading prefixes).
        "redundancy": round(total / estimate_tokens(document), 2),
    }


def _line_at(document: str, pos: int) -> tuple[int, str]:
    start = document.rfind("\n", 0, pos) + 1
    end = document.find("\n", pos)
    return start, document[start : end if end != -1 else len(document)]


def classify_cut(document: str, pos: int) -> str:
    if pos <= 0 or pos >= len(document):
        return "clean"
    if document[pos - 1] == "\n" or document[pos] == "\n":
        return "clean"
    line_start, line = _line_at(document, pos)
    stripped = line.lstrip()
    if stripped.startswith("|"):
        return "table-row"
    if stripped.startswith(("- ", "* ")):
        return "list-item"
    if stripped.startswith("#"):
        return "heading"
    if document[max(line_start, pos - 2) : pos].rstrip(" ").endswith((".", "?", "!")):
        return "sentence-end"
    return "mid-sentence"


def is_mid_word(document: str, pos: int) -> bool:
    return 0 < pos < len(document) and document[pos - 1].isalnum() and document[pos].isalnum()


def _context(document: str, pos: int, width: int = 25) -> str:
    left = document[max(0, pos - width) : pos]
    right = document[pos : pos + width]
    return show(left) + "|" + show(right)


def find_cuts(chunks: list[Chunk], document: str) -> list[Cut]:
    """Classify every interior chunk boundary (document start and end are skipped)."""
    cuts = []
    for c in chunks:
        for side, pos in (("start", c.start_char), ("end", c.end_char)):
            if 0 < pos < len(document.rstrip("\n")):
                cuts.append(
                    Cut(
                        c.id,
                        side,
                        pos,
                        classify_cut(document, pos),
                        is_mid_word(document, pos),
                        _context(document, pos),
                    )
                )
    return cuts


def cut_summary(cuts: list[Cut]) -> dict[str, int]:
    counts = {kind: sum(c.kind == kind for c in cuts) for kind in CUT_KINDS}
    counts["mid-word"] = sum(c.mid_word for c in cuts)
    return counts


def show(text: str) -> str:
    return text.replace("\n", "⏎")


def format_table(chunks: list[Chunk], width: int = 60) -> str:
    rows = [("id", "section", "tok", "starts with", "ends with")]
    for c in chunks:
        section = c.heading_path[-1] if c.heading_path else "—"
        part = f" [{c.meta['part']}]" if "part" in c.meta else ""
        rows.append(
            (
                c.id,
                section[:34] + part,
                str(c.est_tokens),
                show(c.body[:width]),
                show(c.body[-width:]),
            )
        )
    widths = [max(len(r[i]) for r in rows) for i in range(3)]
    lines = []
    for r in rows:
        head = "  ".join(r[i].ljust(widths[i]) for i in range(3))
        lines.append(f"{head}  {r[3]}\n{' ' * len(head)}  … {r[4]}")
    return "\n".join(lines)
