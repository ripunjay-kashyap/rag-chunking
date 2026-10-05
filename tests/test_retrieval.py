import numpy as np
import pytest

from rag_chunking.chunking import Chunk
from rag_chunking.retrieval import Index, select_context


def _chunk(i: int, tokens: int) -> Chunk:
    return Chunk(f"c{i}", "fixed", "x", "x", (), 0, 1, tokens)


def _index(tokens: list[int]) -> Index:
    # Chunk i points along axis i, so the query decides the ranking exactly.
    return Index([_chunk(i, t) for i, t in enumerate(tokens)], np.eye(len(tokens)))


def test_search_ranks_by_cosine_and_breaks_ties_by_document_order():
    index = _index([100] * 4)
    query = np.array([0.1, 0.5, 0.5, 0.9])
    hits = index.search(query)
    assert [h.chunk.id for h in hits] == ["c3", "c1", "c2", "c0"]
    assert [h.rank for h in hits] == [1, 2, 3, 4]
    assert [h.chunk.id for h in index.search(query, n=2)] == ["c3", "c1"]


def test_top_k_mode():
    hits = _index([100] * 5).search(np.array([5, 4, 3, 2, 1.0]))
    assert [h.chunk.id for h in select_context(hits, "top_k", 3, 400)] == ["c0", "c1", "c2"]


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [
        ([150, 150, 150, 50], ["c0", "c1"]),  # 300 fits, the third would make 450
        ([150, 300, 50, 50], ["c0"]),  # stops at the first misfit, no skipping ahead
        ([500, 100, 100, 100], ["c0"]),  # always at least one chunk
        ([100, 100, 100, 100], ["c0", "c1", "c2", "c3"]),  # exactly at the budget
    ],
)
def test_token_budget_mode(tokens, expected):
    hits = _index(tokens).search(np.array([4, 3, 2, 1.0]))
    context = select_context(hits, "token_budget", 3, 400)
    assert [h.chunk.id for h in context] == expected


def test_mismatched_lengths():
    with pytest.raises(ValueError):
        Index([_chunk(0, 10)], np.eye(2))
