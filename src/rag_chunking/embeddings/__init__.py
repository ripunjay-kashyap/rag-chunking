from pathlib import Path

from rag_chunking.config import ConfigError, EmbeddingConfig
from rag_chunking.embeddings.base import Embedder
from rag_chunking.embeddings.cache import EmbeddingCache
from rag_chunking.embeddings.gemini import GeminiEmbedder


def make_embedder(config: EmbeddingConfig, cache_dir: Path) -> GeminiEmbedder:
    if config.provider != "gemini":
        raise ConfigError(f"unsupported embedding provider '{config.provider}'")
    return GeminiEmbedder(config, EmbeddingCache(cache_dir, config.model, config.dimensions))


__all__ = ["Embedder", "EmbeddingCache", "GeminiEmbedder", "make_embedder"]
