"""OpenAI-compatible backend: Groq (hosted open models) or Ollama (local)."""

from __future__ import annotations

from rag_chunking.config import LLMConfig, require_env
from rag_chunking.embeddings.throttle import with_retries


def _is_retryable(exc: Exception) -> bool:
    import openai

    if isinstance(exc, openai.RateLimitError | openai.APIConnectionError):
        return True
    return isinstance(exc, openai.APIStatusError) and exc.status_code >= 500


class OpenAICompatBackend:
    def __init__(self, config: LLMConfig) -> None:
        self.config = config
        self._client = None

    def complete(self, system: str, user: str) -> str:
        import openai

        if self._client is None:
            # Ollama ignores the key, but the client requires a non-empty one.
            key = require_env(self.config.api_key_env) if self.config.api_key_env else "unused"
            self._client = openai.OpenAI(base_url=self.config.base_url, api_key=key)
        response = with_retries(
            lambda: self._client.chat.completions.create(
                model=self.config.model,
                temperature=self.config.temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            ),
            _is_retryable,
        )
        return response.choices[0].message.content or ""
