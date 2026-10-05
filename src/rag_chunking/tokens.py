"""Model-independent token estimate.

Chunk boundaries must not move when the embedding or generation model is swapped, so no
real tokenizer is used. ~4 characters per token is the usual English approximation.
"""

from __future__ import annotations

import math

CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    return math.ceil(len(text) / CHARS_PER_TOKEN)
