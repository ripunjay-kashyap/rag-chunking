"""Gemini backend (google-genai)."""

from __future__ import annotations

from rag_chunking.config import LLMConfig, require_env
from rag_chunking.embeddings.gemini import is_retryable
from rag_chunking.embeddings.throttle import with_retries


class GeminiBackend:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = None

    def complete(self, system: str, user: str) -> str:
        from google import genai
        from google.genai import types

        if self._client is None:
            self._client = genai.Client(api_key=require_env(self.config.api_key_env))
        # Thinking is left at the model default ("minimal" is not supported by every model).
        cfg = types.GenerateContentConfig(
            temperature=self.config.temperature, system_instruction=system
        )
        response = with_retries(
            lambda: self._client.models.generate_content(
                model=self.config.model, contents=user, config=cfg
            ),
            is_retryable,
        )
        if not response.text:
            raise RuntimeError(f"empty response from {self.config.model}: {response}")
        return response.text
