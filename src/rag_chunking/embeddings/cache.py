"""Disk cache for embeddings: a re-run with a warm cache makes no API calls."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np


class EmbeddingCache:
    """One .npy file per vector under <root>/embeddings/<model>-<dims>/.

    The key hashes the model, the dimensions and the exact text sent to the API. The
    text already carries the document/query template, so the retrieval intent is part
    of the key without a separate field.
    """

    def __init__(self, root: Path, model_id: str, dimensions: int) -> None:
        self.model_id = model_id
        self.dimensions = dimensions
        safe = "".join(ch if ch.isalnum() or ch in "-._" else "_" for ch in model_id)
        self.dir = root / "embeddings" / f"{safe}-{dimensions}"
        self.hits = 0
        self.misses = 0

    def _path(self, text: str) -> Path:
        key = hashlib.sha256(f"{self.model_id}|{self.dimensions}|{text}".encode()).hexdigest()
        return self.dir / f"{key}.npy"

    def get(self, text: str) -> np.ndarray | None:
        path = self._path(text)
        if path.is_file():
            self.hits += 1
            return np.load(path)
        self.misses += 1
        return None

    def put(self, text: str, vector: np.ndarray) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self._path(text)
        # Write-then-rename so an interrupted run never leaves a truncated vector.
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        with tmp.open("wb") as fh:
            np.save(fh, vector.astype(np.float32))
        tmp.replace(path)
