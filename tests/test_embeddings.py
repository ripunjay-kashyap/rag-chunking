from types import SimpleNamespace

import numpy as np
import pytest
from google.genai import errors

from rag_chunking.config import load_config
from rag_chunking.embeddings import EmbeddingCache, GeminiEmbedder
from rag_chunking.embeddings.gemini import is_retryable
from rag_chunking.embeddings.throttle import RateLimiter, RetryError, with_retries

CONFIG = load_config().embedding


class FakeClient:
    """Mimics client.models.embed_content: one vector per Content, deterministic values."""

    def __init__(self, dims: int):
        self.dims = dims
        self.requests: list[list[str]] = []
        self.models = self

    def embed_content(self, model, contents, config):
        texts = [c.parts[0].text for c in contents]
        self.requests.append(texts)
        embeddings = []
        for t in texts:
            rng = np.random.default_rng(abs(hash(t)) % 2**32)
            embeddings.append(SimpleNamespace(values=(rng.random(self.dims) * 3).tolist()))
        return SimpleNamespace(embeddings=embeddings)


def _embedder(tmp_path, client=None):
    cache = EmbeddingCache(tmp_path, CONFIG.model, CONFIG.dimensions)
    return GeminiEmbedder(CONFIG, cache, client or FakeClient(CONFIG.dimensions))


def test_one_vector_per_text_normalized(tmp_path):
    embedder = _embedder(tmp_path)
    vectors = embedder.embed_documents(["alpha", "beta", "gamma"])
    assert vectors.shape == (3, CONFIG.dimensions)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1)
    assert embedder.api_calls == 1


def test_templates_applied_and_braces_survive(tmp_path):
    client = FakeClient(CONFIG.dimensions)
    embedder = _embedder(tmp_path, client)
    embedder.embed_documents(["set {x} = 1"])
    embedder.embed_query("what is {x}?")
    assert client.requests == [
        ["title: none | text: set {x} = 1"],
        ["task: search result | query: what is {x}?"],
    ]


def test_warm_cache_makes_no_calls(tmp_path):
    first = _embedder(tmp_path)
    cold = first.embed_documents(["alpha", "beta"])
    second = _embedder(tmp_path)  # fresh process, same cache dir
    warm = second.embed_documents(["alpha", "beta"])
    assert second.api_calls == 0 and second.cache.hits == 2
    assert np.array_equal(cold, warm)


def test_document_and_query_of_same_text_are_cached_separately(tmp_path):
    embedder = _embedder(tmp_path)
    embedder.embed_documents(["same"])
    embedder.embed_query("same")
    assert embedder.api_calls == 2


def test_batches_and_dedupes(tmp_path):
    client = FakeClient(CONFIG.dimensions)
    embedder = _embedder(tmp_path, client)
    texts = [f"t{i}" for i in range(250)] + ["t0"]
    vectors = embedder.embed_documents(texts)
    assert [len(r) for r in client.requests] == [100, 100, 50]
    assert np.array_equal(vectors[0], vectors[-1])


def test_retry_then_give_up():
    sleeps = []
    flaky = iter([errors.APIError(429, {}), errors.APIError(503, {})])

    def call():
        exc = next(flaky, None)
        if exc:
            raise exc
        return "ok"

    assert with_retries(call, is_retryable, sleep=sleeps.append) == "ok"
    assert len(sleeps) == 2

    def always_429():
        raise errors.APIError(429, {})

    with pytest.raises(RetryError, match="gave up after 2"):
        with_retries(always_429, is_retryable, max_retries=2, sleep=sleeps.append)

    def bad_request():
        raise errors.APIError(400, {})

    with pytest.raises(errors.APIError):  # not retried
        with_retries(bad_request, is_retryable, sleep=lambda s: pytest.fail("slept"))

    daily = errors.APIError(429, {"error": {"message": "quotaId: RequestsPerDayPerProject"}})
    assert not is_retryable(daily)


def test_rate_limiter_waits_for_window():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    limiter = RateLimiter(2, 1000, clock=lambda: now[0], sleep=sleep)
    limiter.acquire(10)
    limiter.acquire(10)
    limiter.acquire(10)  # third request in the same minute must wait
    assert slept and now[0] >= 60
    with pytest.raises(ValueError):
        limiter.acquire(5000)
