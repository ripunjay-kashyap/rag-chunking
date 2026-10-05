"""LLM interface and the generation cache shared by every backend."""

from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from rag_chunking.config import LLMConfig
from rag_chunking.embeddings.throttle import RateLimiter


@dataclass(frozen=True)
class Generation:
    text: str
    latency_s: float  # of the original API call; a cache hit returns the stored value


class LLM(Protocol):
    @property
    def model_id(self) -> str: ...

    def generate(self, system: str, user: str) -> Generation: ...


class GenerationCache:
    """One JSON file per prompt under <root>/generations/."""

    def __init__(self, root: Path) -> None:
        self.dir = root / "generations"
        self.hits = 0
        self.misses = 0

    def key(self, config: LLMConfig, system: str, user: str) -> str:
        raw = "|".join([config.provider, config.model, str(config.temperature), system, user])
        return hashlib.sha256(raw.encode()).hexdigest()

    def get(self, key: str) -> Generation | None:
        path = self.dir / f"{key}.json"
        if not path.is_file():
            self.misses += 1
            return None
        self.hits += 1
        data = json.loads(path.read_text(encoding="utf-8"))
        return Generation(data["text"], data["latency_s"])

    def put(self, key: str, generation: Generation, config: LLMConfig) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path = self.dir / f"{key}.json"
        tmp = path.with_suffix(f".{os.getpid()}.tmp")
        payload = {"model": config.model, **asdict(generation)}
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)


class CachedLLM:
    """Wraps a backend's raw call with the cache, rate limiter and call counter."""

    def __init__(self, config: LLMConfig, cache: GenerationCache, backend) -> None:
        self.config = config
        self.cache = cache
        self.backend = backend
        self.limiter = RateLimiter(config.requests_per_minute, tokens_per_minute=10**9)
        self.api_calls = 0

    @property
    def model_id(self) -> str:
        return self.config.model

    def generate(self, system: str, user: str) -> Generation:
        key = self.cache.key(self.config, system, user)
        cached = self.cache.get(key)
        if cached is not None:
            return cached
        self.limiter.acquire(0)
        start = time.perf_counter()
        text = self.backend.complete(system, user)
        generation = Generation(text.strip(), round(time.perf_counter() - start, 2))
        self.api_calls += 1
        self.cache.put(key, generation, self.config)
        return generation
