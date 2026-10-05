"""Strategy A: blind fixed-size character windows.

Truly blind on purpose: no separators, no whitespace snapping, no heading prefix. Cuts
land mid-word, mid-table and mid-list, which is exactly the behaviour being measured.
(LangChain's CharacterTextSplitter splits on "\\n\\n" by default, so it isn't blind.)
"""

from __future__ import annotations

from rag_chunking.chunking.base import Chunk
from rag_chunking.tokens import estimate_tokens


class FixedSizeChunker:
    def __init__(self, size_chars: int, overlap_chars: int) -> None:
        if not 0 <= overlap_chars < size_chars:
            raise ValueError("need 0 <= overlap_chars < size_chars")
        self.size = size_chars
        self.overlap = overlap_chars

    def chunk(self, document: str) -> list[Chunk]:
        if not document:
            return []
        step = self.size - self.overlap
        chunks = []
        start = 0
        while True:
            end = min(start + self.size, len(document))
            body = document[start:end]
            chunks.append(
                Chunk(
                    id=f"A-{len(chunks):03d}",
                    strategy="fixed",
                    text=body,
                    body=body,
                    heading_path=(),
                    start_char=start,
                    end_char=end,
                    est_tokens=estimate_tokens(body),
                )
            )
            # The short tail chunk is kept as is: rebalancing would be a form of
            # boundary awareness.
            if end == len(document):
                return chunks
            start += step
