from pathlib import Path

import pytest

from rag_chunking.__main__ import main
from rag_chunking.chunking import FixedSizeChunker, StructureChunker
from rag_chunking.inspection import classify_cut, find_cuts, is_mid_word, summary_stats
from rag_chunking.normalize import normalize

DOC = normalize(Path("data/diabetes_reference_document.md").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("| a | b |\n| 1 | 2 |\n", "table-row"),
        ("- first item here\n", "list-item"),
        ("## A heading here\n", "heading"),
        ("One sentence. Another one.\n", "sentence-end"),
        ("Words in the middle of prose\n", "mid-sentence"),
    ],
)
def test_classify_cut(text, kind):
    pos = {"table-row": 13, "sentence-end": 14}.get(kind, 8)
    assert classify_cut(text, pos) == kind


def test_line_break_is_clean_and_mid_word_is_a_flag():
    assert classify_cut("line one\nline two", 9) == "clean"
    assert is_mid_word("diagnosis", 4)
    assert not is_mid_word("two words", 3)


def test_structure_boundaries_are_all_clean_fixed_are_not():
    b_cuts = find_cuts(StructureChunker(100, 512, 50, True).chunk(DOC), DOC)
    assert {c.kind for c in b_cuts} == {"clean"}
    a_cuts = find_cuts(FixedSizeChunker(500, 50).chunk(DOC), DOC)
    assert any(c.kind == "table-row" for c in a_cuts)
    assert sum(c.mid_word for c in a_cuts) > 10


def test_summary_stats():
    stats = summary_stats(FixedSizeChunker(500, 50).chunk(DOC), DOC)
    assert stats["chunks"] == 17 and stats["max_tokens"] == 125
    assert stats["redundancy"] > 1.0  # overlap stores some text twice


def test_chunk_command_is_deterministic(tmp_path, capsys):
    config = Path("configs/experiments.toml").read_text(encoding="utf-8")
    config = config.replace('results_dir = "results"', f'results_dir = "{tmp_path}"')
    cfg = tmp_path / "experiments.toml"
    cfg.write_text(config, encoding="utf-8")
    assert main(["--config", str(cfg), "chunk", "--run", "B-min100"]) == 0
    first = (tmp_path / "B-min100" / "chunks.json").read_bytes()
    assert main(["--config", str(cfg), "chunk", "--run", "B-min100"]) == 0
    assert (tmp_path / "B-min100" / "chunks.json").read_bytes() == first
    assert "B-6-complications" in capsys.readouterr().out
