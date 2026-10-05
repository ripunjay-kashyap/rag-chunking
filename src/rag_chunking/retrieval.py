"""Cosine search over a NumPy matrix, with top-k and equal-token-budget context selection.

~30 vectors per run: a vector database would add nothing but opacity at this scale.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from rag_chunking.chunking import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int  # 1-based


class Index:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"{len(chunks)} chunks but {len(vectors)} vectors")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)

    def search(self, query_vector: np.ndarray, n: int | None = None) -> list[Hit]:
        # Vectors are unit-norm, so the dot product is the cosine similarity.
        scores = self.vectors @ query_vector.astype(np.float32)
        # Ties break on document order, so rankings never depend on sort stability.
        order = sorted(range(len(self.chunks)), key=lambda i: (-float(scores[i]), i))
        return [
            Hit(self.chunks[i], float(scores[i]), rank) for rank, i in enumerate(order[:n], start=1)
        ]


def select_context(
    hits: list[Hit],
    mode: Literal["top_k", "token_budget"],
    k: int,
    token_budget: int,
) -> list[Hit]:
    """The hits that are passed to the LLM and scored."""
    if mode == "top_k":
        return hits[:k]
    context: list[Hit] = []
    total = 0
    for hit in hits:
        # Stop at the first chunk that doesn't fit rather than skipping it for a smaller
        # one further down: skipping would reward small chunks over relevant ones.
        if context and total + hit.chunk.est_tokens > token_budget:
            break
        context.append(hit)
        total += hit.chunk.est_tokens
    return context
