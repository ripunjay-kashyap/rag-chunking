"""Embedder interface: swap the model without touching chunking or retrieval."""

from __future__ import annotations

from typing import Protocol

import numpy as np


class Embedder(Protocol):
    @property
    def model_id(self) -> str: ...

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """One L2-normalized row per text, shape (len(texts), dimensions)."""
        ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)
