from pathlib import Path

from rag_chunking.config import LLMConfig
from rag_chunking.generation.base import LLM, CachedLLM, Generation, GenerationCache
from rag_chunking.generation.gemini import GeminiBackend
from rag_chunking.generation.openai_compat import OpenAICompatBackend


def make_llm(config: LLMConfig, cache_dir: Path) -> CachedLLM:
    backend = GeminiBackend(config) if config.provider == "gemini" else OpenAICompatBackend(config)
    return CachedLLM(config, GenerationCache(cache_dir), backend)


__all__ = ["LLM", "CachedLLM", "Generation", "GenerationCache", "make_llm"]
