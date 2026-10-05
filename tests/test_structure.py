from pathlib import Path

import pytest

from rag_chunking.chunking.structure import StructureChunker, parse_sections
from rag_chunking.normalize import normalize

DOC = normalize(Path("data/diabetes_reference_document.md").read_text(encoding="utf-8"))
TITLE = "Type 2 Diabetes Mellitus: A Clinical and Patient Guide"
TABLE_LINES = [line for line in DOC.splitlines() if line.startswith("|")]


def _chunk(min_tokens=100, max_tokens=512, overlap=50, prefix=True, doc=DOC):
    return StructureChunker(min_tokens, max_tokens, overlap, prefix).chunk(doc)


def test_min100_merges_small_siblings():
    chunks = _chunk(min_tokens=100)
    assert [c.id for c in chunks] == [
        "B-1-overview",
        "B-2-epidemiology-and-risk-factors",
        "B-3-symptoms",
        "B-4-diagnostic-criteria",
        "B-5-1-lifestyle-modifications",
        "B-5-treatment-approaches",
        "B-6-complications",
        "B-7-monitoring-and-follow-up",
        "B-8-self-management-and-lifestyle-tips",
        "B-9-when-to-seek-emergency-care",
        "B-10-frequently-asked-questions",
    ]
    by_id = {c.id: c for c in chunks}
    six = by_id["B-6-complications"]
    # The merged chunk speaks for the parent; sub-headings stay in the body.
    assert six.heading_path == (TITLE, "6. Complications")
    assert six.meta["merged_from"] == [
        "6.1 Microvascular Complications",
        "6.2 Macrovascular Complications",
    ]
    assert six.body.startswith("### 6.1 Microvascular") and "### 6.2 Macrovascular" in six.body
    # 5.3 is small and merges into its previous sibling 5.2, not into 5.1 or §6.
    assert by_id["B-5-treatment-approaches"].meta["merged_from"] == [
        "5.2 Oral Medications",
        "5.3 Insulin Therapy",
    ]


def test_min40_keeps_every_section():
    chunks = _chunk(min_tokens=40)
    assert len(chunks) == 13
    assert all(len(c.heading_path) >= 2 for c in chunks)


def test_invariants_on_real_document():
    for chunks in (_chunk(min_tokens=100), _chunk(min_tokens=40)):
        for c in chunks:
            assert c.body.strip()
            assert c.heading_path and c.heading_path[0] == TITLE
            assert DOC[c.start_char : c.end_char] == c.body
            assert c.text == " > ".join(c.heading_path) + "\n\n" + c.body
            assert c.est_tokens <= 512
        # Empty parents never become chunks of their own.
        assert all(c.heading_path != (TITLE,) for c in chunks)
        # The §4 table is whole in exactly one chunk.
        assert sum(all(line in c.body for line in TABLE_LINES) for c in chunks) == 1


def test_prefix_toggle_changes_text_not_boundaries():
    with_prefix, without = _chunk(prefix=True), _chunk(prefix=False)
    assert [(c.start_char, c.end_char) for c in with_prefix] == [
        (c.start_char, c.end_char) for c in without
    ]
    assert all(c.text == c.body for c in without)
    assert all(a.est_tokens > b.est_tokens for a, b in zip(with_prefix, without, strict=True))


def test_fallback_triggers_only_on_large_sections():
    chunks = _chunk(min_tokens=40, max_tokens=180, overlap=18)
    split = sorted({c.heading_path[-1] for c in chunks if "part" in c.meta})
    assert split == [
        "10. Frequently Asked Questions",
        "2. Epidemiology and Risk Factors",
        "4. Diagnostic Criteria",
        "5.2 Oral Medications",
    ]
    assert all(c.est_tokens <= 180 for c in chunks)
    # Every piece keeps the full heading path.
    assert all(c.text.startswith(" > ".join(c.heading_path)) for c in chunks)


def test_oversized_table_repeats_header():
    chunks = _chunk(min_tokens=40, max_tokens=100, overlap=10)
    table_pieces = [c for c in chunks if "| Test | Normal" in c.body]
    assert len(table_pieces) == 2
    assert table_pieces[1].meta["repeated_header"] is True
    for piece in table_pieces:
        assert piece.body.startswith(TABLE_LINES[0] + "\n" + TABLE_LINES[1])
    rows = [line for c in table_pieces for line in c.body.splitlines()[2:]]
    assert rows == TABLE_LINES[2:]
    assert all(c.est_tokens <= 100 for c in chunks)


def test_hash_inside_code_is_not_a_heading():
    doc = "# T\n\n## A\n\n```bash\n# not a heading\necho hi\n```\n\n## B\n\nText.\n"
    titles = [s.title for s in parse_sections(doc)]
    assert titles == ["T", "A", "B"]


def test_never_merges_across_parents():
    doc = "# T\n\n## A\n\n### a1\n\nshort a\n\n## B\n\n### b1\n\nshort b\n"
    chunks = _chunk(min_tokens=100, doc=doc)
    assert [c.heading_path for c in chunks] == [("T", "A", "a1"), ("T", "B", "b1")]
    assert all("merged_from" not in c.meta for c in chunks)


def test_run_of_small_siblings_merges_together():
    doc = "# T\n\n## P\n\n### x\n\none\n\n### y\n\ntwo\n\n### z\n\nthree\n\n## Q\n\nother\n"
    chunks = _chunk(min_tokens=100, doc=doc)
    assert chunks[0].heading_path == ("T", "P")
    assert chunks[0].meta["merged_from"] == ["x", "y", "z"]
    assert chunks[1].heading_path == ("T", "Q")


def test_ids_are_unique_and_stable():
    first, second = _chunk(), _chunk()
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == len(first)


def test_invalid_min_max():
    with pytest.raises(ValueError):
        StructureChunker(512, 512, 0, True)
