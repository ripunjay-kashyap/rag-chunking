from dataclasses import replace

from rag_chunking.chunking import Chunk
from rag_chunking.config import load_config
from rag_chunking.generation import CachedLLM, GenerationCache, make_llm
from rag_chunking.generation.gemini import GeminiBackend
from rag_chunking.generation.openai_compat import OpenAICompatBackend
from rag_chunking.generation.prompts import REFUSAL, SYSTEM_PROMPT, user_prompt

CONFIG = load_config()


class FakeBackend:
    def __init__(self):
        self.calls = 0

    def complete(self, system, user):
        self.calls += 1
        return f"  answer #{self.calls}  "


def _chunk(cid: str, text: str) -> Chunk:
    return Chunk(cid, "structure", text, text, (), 0, 1, 10)


def test_prompt_numbers_sources_and_hides_chunk_ids():
    prompt = user_prompt(
        "Q?", [_chunk("B-4-diagnostic-criteria", "first"), _chunk("A-007", "second")]
    )
    assert prompt == "Sources:\n\n[1]\nfirst\n\n[2]\nsecond\n\nQuestion: Q?"
    assert "diagnostic" not in prompt and "A-007" not in prompt
    assert REFUSAL in SYSTEM_PROMPT


def test_cache_returns_stored_answer_without_calling(tmp_path):
    backend = FakeBackend()
    llm = CachedLLM(CONFIG.llm, GenerationCache(tmp_path), backend)
    first = llm.generate("sys", "user")
    assert first.text == "answer #1" and llm.api_calls == 1

    fresh = CachedLLM(CONFIG.llm, GenerationCache(tmp_path), FakeBackend())
    again = fresh.generate("sys", "user")
    assert again == first and fresh.api_calls == 0


def test_cache_key_depends_on_model_and_prompt(tmp_path):
    cache = GenerationCache(tmp_path)
    base = cache.key(CONFIG.llm, "sys", "user")
    assert base != cache.key(CONFIG.llm, "sys", "user2")
    assert base != cache.key(replace(CONFIG.llm, model="other"), "sys", "user")
    assert base != cache.key(replace(CONFIG.llm, temperature=0.7), "sys", "user")


def test_backend_is_chosen_by_config_only(tmp_path):
    assert isinstance(make_llm(CONFIG.llm_profiles["gemini"], tmp_path).backend, GeminiBackend)
    for name in ("groq", "ollama"):
        llm = make_llm(CONFIG.llm_profiles[name], tmp_path)
        assert isinstance(llm.backend, OpenAICompatBackend)
        assert llm.model_id == CONFIG.llm_profiles[name].model
