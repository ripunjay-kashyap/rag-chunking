"""Chunk data model shared by both strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class Chunk:
    id: str
    strategy: str
    text: str  # exactly what gets embedded and shown to the LLM
    # Text without the heading prefix. Equals document[start_char:end_char], except for
    # fallback pieces with meta["repeated_header"] (table header / code fence re-added).
    body: str
    heading_path: tuple[str, ...]  # empty for Strategy A
    start_char: int
    end_char: int
    est_tokens: int  # of `text`, since that is what costs context
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "strategy": self.strategy,
            "heading_path": list(self.heading_path),
            "start_char": self.start_char,
            "end_char": self.end_char,
            "est_tokens": self.est_tokens,
            "meta": self.meta,
            "text": self.text,
            "body": self.body,
        }


class Chunker(Protocol):
    def chunk(self, document: str) -> list[Chunk]: ...
