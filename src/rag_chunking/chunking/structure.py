"""Strategy B: one chunk per Markdown section, with a heading-path prefix.

Order of operations: parse sections -> merge small ones -> split oversized ones.
Fallback pieces are never merged again.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rag_chunking.chunking.base import Chunk
from rag_chunking.chunking.recursive import split_body
from rag_chunking.tokens import CHARS_PER_TOKEN, estimate_tokens

_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
_FENCE = re.compile(r"^\s*(```|~~~)")
PATH_SEPARATOR = " > "


@dataclass(frozen=True)
class Section:
    level: int
    title: str
    path: tuple[str, ...]
    heading_start: int
    body_start: int  # body excludes the heading line and surrounding blank lines
    body_end: int

    @property
    def parent(self) -> tuple[str, ...]:
        return self.path[:-1]


def parse_sections(document: str) -> list[Section]:
    """Every heading in document order. `#` lines inside fenced code are not headings."""
    headings: list[tuple[int, str, int, int]] = []  # level, title, line start, line end
    fence: str | None = None
    pos = 0
    for line in document.splitlines(keepends=True):
        fence_match = _FENCE.match(line)
        if fence_match:
            if fence is None:
                fence = fence_match.group(1)
            elif fence_match.group(1) == fence:
                fence = None
        elif fence is None:
            m = _HEADING.match(line.rstrip("\n"))
            if m:
                headings.append((len(m.group(1)), m.group(2), pos, pos + len(line)))
        pos += len(line)

    sections = []
    stack: list[tuple[int, str]] = []
    for i, (level, title, start, line_end) in enumerate(headings):
        stack = [s for s in stack if s[0] < level] + [(level, title)]
        end = headings[i + 1][2] if i + 1 < len(headings) else len(document)
        body_start, body_end = line_end, end
        while body_start < body_end and document[body_start].isspace():
            body_start += 1
        while body_end > body_start and document[body_end - 1].isspace():
            body_end -= 1
        sections.append(
            Section(level, title, tuple(t for _, t in stack), start, body_start, body_end)
        )
    return sections


@dataclass(frozen=True)
class _Unit:
    """One or more adjacent sibling sections that become one chunk (before any split)."""

    sections: tuple[Section, ...]

    @property
    def parent(self) -> tuple[str, ...]:
        return self.sections[0].parent

    @property
    def path(self) -> tuple[str, ...]:
        # A merged chunk speaks for the parent; each sub-heading stays in its body.
        return self.sections[0].path if len(self.sections) == 1 else self.parent

    @property
    def start(self) -> int:
        first = self.sections[0]
        return first.body_start if len(self.sections) == 1 else first.heading_start

    @property
    def end(self) -> int:
        return self.sections[-1].body_end


def merge_small(units: list[_Unit], document: str, min_tokens: int) -> list[_Unit]:
    """Merge units under min_tokens into an adjacent sibling; previous first, then next.

    Measured on the body, not the prefixed text: the min rule filters thin *content*,
    and measuring the prefix would make the result depend on the prefix toggle.
    """

    def small(u: _Unit) -> bool:
        return estimate_tokens(document[u.start : u.end]) < min_tokens

    units = list(units)
    stuck: set[int] = set()  # small units with no sibling to merge into
    while True:
        i = next((i for i, u in enumerate(units) if small(u) and i not in stuck), None)
        if i is None:
            return units
        if i > 0 and units[i - 1].parent == units[i].parent:
            j = i - 1
        elif i + 1 < len(units) and units[i + 1].parent == units[i].parent:
            j = i
        else:
            stuck.add(i)
            continue
        units[j : j + 2] = [_Unit(units[j].sections + units[j + 1].sections)]
        # Indices shifted; recompute which units are genuinely stuck.
        stuck = set()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


class StructureChunker:
    def __init__(self, min_tokens: int, max_tokens: int, overlap_tokens: int, prefix: bool) -> None:
        if not 0 <= min_tokens < max_tokens:
            raise ValueError("need 0 <= min_tokens < max_tokens")
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.prefix = prefix

    def _header(self, path: tuple[str, ...]) -> str:
        return PATH_SEPARATOR.join(path) + "\n\n" if self.prefix else ""

    def chunk(self, document: str) -> list[Chunk]:
        # Empty parents (the title, "5. Treatment Approaches") produce no chunk;
        # their titles live on in their children's paths.
        units = [_Unit((s,)) for s in parse_sections(document) if s.body_end > s.body_start]
        units = merge_small(units, document, self.min_tokens)

        chunks: list[Chunk] = []
        seen_ids: dict[str, int] = {}
        for unit in units:
            header = self._header(unit.path)
            body = document[unit.start : unit.end]
            meta: dict[str, Any] = {}
            if len(unit.sections) > 1:
                meta["merged_from"] = [s.title for s in unit.sections]

            # The ceiling applies to the final text: that is what the embedder sees.
            if estimate_tokens(header + body) <= self.max_tokens:
                parts = [(body, unit.start, unit.end, False)]
            else:
                budget = self.max_tokens * CHARS_PER_TOKEN - len(header)
                overlap = self.overlap_tokens * CHARS_PER_TOKEN
                parts = [
                    (p.text, unit.start + p.start, unit.start + p.end, p.repeated_header)
                    for p in split_body(body, budget, overlap)
                ]

            base = "B-" + _slug(unit.path[-1])
            seen_ids[base] = seen_ids.get(base, 0) + 1
            if seen_ids[base] > 1:
                base = f"{base}-{seen_ids[base]}"
            for k, (text, start, end, repeated) in enumerate(parts, start=1):
                part_meta = dict(meta)
                if len(parts) > 1:
                    part_meta["part"] = f"{k}/{len(parts)}"
                if repeated:
                    part_meta["repeated_header"] = True
                chunks.append(
                    Chunk(
                        id=f"{base}-p{k}" if len(parts) > 1 else base,
                        strategy="structure",
                        text=header + text,
                        body=text,
                        heading_path=unit.path,
                        start_char=start,
                        end_char=end,
                        est_tokens=estimate_tokens(header + text),
                        meta=part_meta,
                    )
                )
        return chunks
