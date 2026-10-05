from pathlib import Path

import pytest

from rag_chunking.chunking.fixed import FixedSizeChunker
from rag_chunking.normalize import normalize
from rag_chunking.tokens import estimate_tokens

DOC = normalize(Path("data/diabetes_reference_document.md").read_text(encoding="utf-8"))


def _reconstruct(chunks, overlap):
    return chunks[0].body + "".join(c.body[overlap:] for c in chunks[1:])


@pytest.mark.parametrize(
    ("size", "overlap", "count"), [(500, 50, 17), (300, 30, 28), (800, 80, 11)]
)
def test_matrix_sizes(size, overlap, count):
    chunks = FixedSizeChunker(size, overlap).chunk(DOC)
    assert len(chunks) == count
    assert all(len(c.body) <= size for c in chunks)
    assert all(len(c.body) == size for c in chunks[:-1])
    for prev, cur in zip(chunks, chunks[1:], strict=False):
        assert prev.body[-overlap:] == cur.body[:overlap]
    assert _reconstruct(chunks, overlap) == DOC


def test_offsets_and_fields():
    chunks = FixedSizeChunker(500, 50).chunk(DOC)
    for i, c in enumerate(chunks):
        assert c.id == f"A-{i:03d}"
        assert DOC[c.start_char : c.end_char] == c.body == c.text
        assert c.heading_path == ()
        assert c.est_tokens == estimate_tokens(c.text)
    assert chunks[-1].end_char == len(DOC)


def test_blind_cut_ignores_boundaries():
    # The first window ends exactly at char 500, wherever that falls.
    first = FixedSizeChunker(500, 50).chunk(DOC)[0]
    assert first.body == DOC[:500]


def test_short_and_empty_documents():
    assert FixedSizeChunker(10, 2).chunk("") == []
    [only] = FixedSizeChunker(10, 2).chunk("abc")
    assert only.body == "abc"
    # Exactly one window long: no empty or duplicate tail chunk.
    assert len(FixedSizeChunker(10, 2).chunk("x" * 10)) == 1


def test_invalid_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(100, 100)


def test_normalize():
    assert normalize("a  \r\nb\t\n\n\n\n\nc") == "a\nb\n\n\nc\n"
    assert normalize(DOC) == DOC
    assert estimate_tokens("abcde") == 2
