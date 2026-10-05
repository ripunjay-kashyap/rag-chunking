"""The single grounded prompt shared by every run and every backend."""

from __future__ import annotations

from rag_chunking.chunking import Chunk

REFUSAL = "The document does not contain this information."

# The rules are generic grounding rules, not hints about specific questions: they apply
# the same way to both strategies, so they can't favour either one.
SYSTEM_PROMPT = f"""You answer questions using only the numbered sources provided.

Rules:
1. Use only information stated in the sources. Do not use outside knowledge.
2. If the sources do not contain the answer, reply exactly: "{REFUSAL}"
3. If the question assumes something the sources contradict or do not support, say so,
   then answer with what the sources do say.
4. Keep numbers, units and ranges exactly as written in the sources. Do not narrow a
   range to a single value.
5. Cite the sources you used by number, e.g. [1] or [1][3].
6. Be concise."""


def format_context(chunks: list[Chunk]) -> str:
    # Sources are numbered by rank, not labelled with chunk IDs: Strategy B's IDs name
    # the section (e.g. "B-4-diagnostic-criteria"), which would leak heading context.
    return "\n\n".join(f"[{i}]\n{c.text.strip()}" for i, c in enumerate(chunks, start=1))


def user_prompt(question: str, chunks: list[Chunk]) -> str:
    return f"Sources:\n\n{format_context(chunks)}\n\nQuestion: {question}"
