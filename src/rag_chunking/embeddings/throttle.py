"""Client-side rate limiting and retry, shared by every API client."""

from __future__ import annotations

import random
import time
from collections import deque
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


class RateLimiter:
    """Sliding 60-second window over requests and (estimated) tokens."""

    def __init__(
        self,
        requests_per_minute: int,
        tokens_per_minute: int,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.rpm = requests_per_minute
        self.tpm = tokens_per_minute
        self._clock = clock
        self._sleep = sleep
        self._window: deque[tuple[float, int]] = deque()

    def acquire(self, tokens: int) -> None:
        if tokens > self.tpm:
            raise ValueError(f"one request of {tokens} tokens exceeds the {self.tpm} TPM limit")
        while True:
            now = self._clock()
            while self._window and now - self._window[0][0] >= 60:
                self._window.popleft()
            used = sum(t for _, t in self._window)
            if len(self._window) < self.rpm and used + tokens <= self.tpm:
                self._window.append((now, tokens))
                return
            self._sleep(max(0.05, 60 - (now - self._window[0][0])))


class RetryError(RuntimeError):
    pass


def with_retries(
    call: Callable[[], T],
    is_retryable: Callable[[Exception], bool],
    max_retries: int = 5,
    base_delay: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Exponential backoff with full jitter on retryable errors (429, 5xx)."""
    for attempt in range(max_retries + 1):
        try:
            return call()
        except Exception as exc:
            if not is_retryable(exc) or attempt == max_retries:
                if is_retryable(exc):
                    raise RetryError(f"gave up after {max_retries} retries: {exc}") from exc
                raise
            sleep(random.uniform(0, base_delay * 2**attempt))
    raise AssertionError("unreachable")
