"""Recursive fallback splitter for sections over the size ceiling.

Never triggers on the reference document at max_tokens = 512; it is a safety net for
longer documents. Two ideas drive it:

- Tables and fenced code are atomic: a table row without its header is meaningless, so a
  cut may only fall between blocks. A block that alone exceeds the budget is split by
  rows (table, header repeated) or blank lines (code, fence re-opened).
- The piece count is chosen up front, n = ceil(size / budget), and every cut aims at
  i * size / n. This gives even pieces instead of greedy fill + a tiny tail.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

# Coarsest first. A cut position is the index just after the separator.
SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", " ")
_SENTENCE_LEVEL = (". ", "? ", "! ")
# How far from the ideal position a coarser separator may be before a finer one wins.
_TOLERANCE = 0.25
_FENCE = re.compile(r"^(```|~~~)")


@dataclass(frozen=True)
class Block:
    kind: str  # "text" | "table" | "code"
    start: int
    end: int


@dataclass(frozen=True)
class Piece:
    text: str
    start: int  # offsets into the text given to split_body()
    end: int
    repeated_header: bool = False  # text is not a plain slice: header rows were re-added


def find_blocks(text: str) -> list[Block]:
    """Partition text into text / table / code blocks of whole lines."""
    lines = text.splitlines(keepends=True)
    blocks: list[Block] = []
    pos = 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if _FENCE.match(line.lstrip()):
            fence = _FENCE.match(line.lstrip()).group(1)  # type: ignore[union-attr]
            j = i + 1
            while j < len(lines) and not lines[j].lstrip().startswith(fence):
                j += 1
            j = min(j + 1, len(lines))  # include the closing fence (if any)
            kind = "code"
        elif line.lstrip().startswith("|"):
            j = i + 1
            while j < len(lines) and lines[j].lstrip().startswith("|"):
                j += 1
            kind = "table"
        else:
            j = i + 1
            while (
                j < len(lines)
                and not lines[j].lstrip().startswith("|")
                and not _FENCE.match(lines[j].lstrip())
            ):
                j += 1
            kind = "text"
        end = pos + sum(len(line) for line in lines[i:j])
        if blocks and blocks[-1].kind == kind == "text":
            blocks[-1] = Block("text", blocks[-1].start, end)
        else:
            blocks.append(Block(kind, pos, end))
        pos, i = end, j
    return blocks


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _inside(pos: int, atomic: list[Block]) -> Block | None:
    for b in atomic:
        if b.start < pos < b.end:
            return b
    return None


def _candidates(text: str, lo: int, hi: int, seps: tuple[str, ...], atomic: list[Block]):
    out = []
    for sep in seps:
        idx = text.find(sep, lo)
        while idx != -1 and idx + len(sep) < hi:
            pos = idx + len(sep)
            if pos > lo and not _inside(pos, atomic):
                out.append(pos)
            idx = text.find(sep, idx + 1)
    return sorted(set(out))


def _choose_cut(text: str, prev: int, hi: int, target: float, window: float, atomic):
    levels = [(s,) for s in SEPARATORS if s not in _SENTENCE_LEVEL]
    levels.insert(2, _SENTENCE_LEVEL)  # sentence ends are one level, after "\n"
    lo_w, hi_w = max(prev + 1, target - window), min(hi - 1, target + window)
    for seps in levels:
        found = [p for p in _candidates(text, prev, hi, seps, atomic) if lo_w <= p <= hi_w]
        if found:
            return min(found, key=lambda p: (abs(p - target), p))
    # Last resort: a hard cut, moved off any atomic block it would land inside.
    pos = round(target)
    block = _inside(pos, atomic)
    if block:
        pos = block.start if pos - block.start <= block.end - pos else block.end
        if pos <= prev:
            pos = block.end
    return pos


def _overlap_start(text: str, cut: int, overlap: int, floor: int, atomic: list[Block]) -> int:
    """Earliest word boundary within `overlap` chars before the cut, outside atomic blocks."""
    for p in range(max(floor, cut - overlap), cut):
        if (p == 0 or text[p - 1].isspace()) and not text[p].isspace() and not _inside(p, atomic):
            return p
    return cut


def _split_region(text: str, lo: int, hi: int, budget: int, overlap: int, atomic) -> list[Piece]:
    lo, hi = _trim(text, lo, hi)
    if hi - lo <= budget:
        return [Piece(text[lo:hi], lo, hi)]
    n = math.ceil((hi - lo) / max(1, budget - overlap))
    while n <= hi - lo:
        size = (hi - lo) / n
        cuts, prev = [], lo
        for i in range(1, n):
            prev = _choose_cut(text, prev, hi, lo + i * size, size * _TOLERANCE, atomic)
            cuts.append(prev)
        bounds = list(zip([lo, *cuts], [*cuts, hi], strict=True))
        pieces = []
        for i, (s, e) in enumerate(bounds):
            s, e = _trim(text, s, e)
            if i > 0:
                # At most half the previous piece, so a piece never restarts where its
                # predecessor started (possible when overlap is large against the budget).
                floor = (bounds[i - 1][0] + bounds[i - 1][1]) // 2
                with_overlap = _overlap_start(text, s, overlap, floor, atomic)
                # Overlap is a nicety; drop it rather than break the budget.
                if e - with_overlap <= budget:
                    s = with_overlap
            if s < e:
                pieces.append(Piece(text[s:e], s, e))
        if all(len(p.text) <= budget for p in pieces):
            return pieces
        n += 1
    raise ValueError("could not split region within budget")


def _split_table(text: str, block: Block, budget: int) -> list[Piece]:
    lines = text[block.start : block.end].rstrip("\n").split("\n")
    header, rows = lines[:2], lines[2:]
    header_text = "\n".join(header) + "\n"
    row_budget = budget - len(header_text)
    if row_budget <= 0 or any(len(r) + 1 > row_budget for r in rows):
        raise ValueError("table header or a single row exceeds the chunk budget")
    sizes = [len(r) + 1 for r in rows]
    total = sum(sizes)
    n = math.ceil(total / row_budget)
    while True:
        # Each row goes to the group its midpoint falls in: even groups, order kept.
        groups: list[list[int]] = [[] for _ in range(n)]
        filled = 0
        for i, size in enumerate(sizes):
            groups[min(n - 1, int((filled + size / 2) / (total / n)))].append(i)
            filled += size
        groups = [g for g in groups if g]
        if max(sum(sizes[i] for i in g) for g in groups) <= row_budget:
            break
        n += 1
    # Offsets of each row inside `text`, so pieces can point back at the document.
    row_starts, pos = [], block.start + len(header[0]) + 1 + len(header[1]) + 1
    for row in rows:
        row_starts.append(pos)
        pos += len(row) + 1
    pieces = []
    for k, g in enumerate(groups):
        body = "\n".join(rows[i] for i in g)
        start = block.start if k == 0 else row_starts[g[0]]
        end = row_starts[g[-1]] + len(rows[g[-1]])
        pieces.append(Piece(header_text + body, start, end, repeated_header=k > 0))
    return pieces


def _split_code(text: str, block: Block, budget: int) -> list[Piece]:
    lines = text[block.start : block.end].rstrip("\n").split("\n")
    has_close = len(lines) > 1 and _FENCE.match(lines[-1].lstrip())
    open_line, inner = lines[0], lines[1 : -1 if has_close else None]
    close_line = _FENCE.match(open_line.lstrip()).group(1)  # type: ignore[union-attr]
    paragraphs = re.split(r"\n\s*\n", "\n".join(inner))
    room = budget - len(open_line) - len(close_line) - 2
    pieces, current = [], ""
    for para in paragraphs:
        joined = f"{current}\n\n{para}" if current else para
        if current and len(joined) > room:
            pieces.append(current)
            current = para
        else:
            current = joined
    pieces.append(current)
    if any(len(p) > room for p in pieces):
        raise ValueError("a code paragraph exceeds the chunk budget")
    # Re-fenced pieces are not slices; they all point at the whole block.
    return [
        Piece(f"{open_line}\n{p}\n{close_line}", block.start, block.end, repeated_header=True)
        for p in pieces
    ]


def split_body(text: str, budget: int, overlap: int = 0) -> list[Piece]:
    """Split text into pieces of at most `budget` characters."""
    blocks = find_blocks(text)
    atomic = [b for b in blocks if b.kind != "text"]
    pieces: list[Piece] = []
    region_start = 0
    for b in atomic:
        if b.end - b.start <= budget:
            continue
        # An oversized atomic block ends the current region and is split on its own.
        if region_start < b.start:
            pieces += _split_region(text, region_start, b.start, budget, overlap, atomic)
        splitter = _split_table if b.kind == "table" else _split_code
        pieces += splitter(text, b, budget)
        region_start = b.end
    if region_start < len(text):
        pieces += _split_region(text, region_start, len(text), budget, overlap, atomic)
    return [p for p in pieces if p.text.strip()]
