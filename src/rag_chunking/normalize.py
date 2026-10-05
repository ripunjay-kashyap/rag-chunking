"""Whitespace-only cleanup of the source document.

Deliberately minimal: production pipelines can't hand-edit documents, and anything more
(e.g. rewriting Markdown) would change what both strategies see. Offsets in every chunk
refer to the output of this function.
"""

from __future__ import annotations

import re

_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_EXCESS_BLANK_LINES = re.compile(r"\n{4,}")


def normalize(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_SPACE.sub("", text)
    # 3+ blank lines (4+ newlines) become 2 blank lines.
    text = _EXCESS_BLANK_LINES.sub("\n\n\n", text)
    return text.rstrip("\n") + "\n"
