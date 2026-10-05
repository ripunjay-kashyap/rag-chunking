"""Gemini embeddings through google-genai, behind the cache and the rate limiter."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from rag_chunking.config import EmbeddingConfig, require_env
from rag_chunking.embeddings.base import l2_normalize
from rag_chunking.embeddings.cache import EmbeddingCache
from rag_chunking.embeddings.throttle import RateLimiter, with_retries
from rag_chunking.tokens import estimate_tokens

log = logging.getLogger(__name__)


def is_retryable(exc: Exception) -> bool:
    from google.genai import errors

    if not isinstance(exc, errors.APIError):
        return False
    # A per-day quota resets in hours: retrying only burns time, so fail fast.
    if exc.code == 429 and "PerDay" in str(exc):
        return False
    return exc.code == 429 or exc.code >= 500


class GeminiEmbedder:
    def __init__(self, config: EmbeddingConfig, cache: EmbeddingCache, client: Any = None):
        self.config = config
        self.cache = cache
        self.limiter = RateLimiter(config.requests_per_minute, config.tokens_per_minute)
        self._client = client  # created lazily: a warm cache needs no API key
        self.api_calls = 0

    @property
    def model_id(self) -> str:
        return self.config.model

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed([_apply(self.config.document_template, t) for t in texts])

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed([_apply(self.config.query_template, t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]

    def _embed(self, texts: list[str]) -> np.ndarray:
        vectors: dict[str, np.ndarray] = {}
        missing = []
        for text in dict.fromkeys(texts):  # dedupe, keep order
            cached = self.cache.get(text)
            if cached is None:
                missing.append(text)
            else:
                vectors[text] = cached
        for i in range(0, len(missing), self.config.batch_size):
            batch = missing[i : i + self.config.batch_size]
            for text, vector in zip(batch, self._call_api(batch), strict=True):
                self.cache.put(text, vector)
                vectors[text] = vector
        if missing:
            log.info("embeddings: %d cached, %d fetched", len(texts) - len(missing), len(missing))
        out = np.stack([vectors[t] for t in texts]) if texts else np.empty((0, 0))
        return out.astype(np.float32)

    def _call_api(self, batch: list[str]) -> np.ndarray:
        from google.genai import types

        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=require_env(self.config.api_key_env))
        # One Content per text. Passing a list of plain strings makes the API return a
        # single embedding for the whole list.
        contents = [types.Content(parts=[types.Part(text=t)]) for t in batch]
        cfg = types.EmbedContentConfig(output_dimensionality=self.config.dimensions)
        self.limiter.acquire(sum(estimate_tokens(t) for t in batch))
        response = with_retries(
            lambda: self._client.models.embed_content(
                model=self.config.model, contents=contents, config=cfg
            ),
            is_retryable,
        )
        self.api_calls += 1
        vectors = np.array([e.values for e in response.embeddings], dtype=np.float32)
        if vectors.shape != (len(batch), self.config.dimensions):
            raise RuntimeError(
                f"expected {len(batch)} x {self.config.dimensions} embeddings, got {vectors.shape}"
            )
        # Normalized here so cosine similarity is a plain dot product for any model.
        return l2_normalize(vectors)


def _apply(template: str, text: str) -> str:
    # Not str.format: chunk text may contain braces.
    return template.replace("{text}", text)
