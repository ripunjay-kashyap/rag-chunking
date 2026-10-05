from rag_chunking.chunking.recursive import find_blocks, split_body

PARAGRAPH = "Sentence one is here. Sentence two follows it. " * 4  # ~190 chars


def test_long_section_splits_at_paragraphs_with_even_sizes():
    paragraphs = [f"{i}. {PARAGRAPH}".strip() for i in range(32)]
    text = "\n\n".join(paragraphs)  # ~6,200 chars ≈ 1,550 tokens
    pieces = split_body(text, budget=2048, overlap=0)
    assert len(pieces) == 4
    assert all(len(p.text) <= 2048 for p in pieces)
    for p in pieces:
        assert p.text.startswith(tuple(f"{i}. " for i in range(32)))
        assert p.end == len(text) or text[p.end : p.end + 2] == "\n\n"
    sizes = [len(p.text) for p in pieces]
    assert min(sizes) / max(sizes) > 0.8  # no tiny tail


def test_overlap_repeats_text_between_pieces():
    text = "\n\n".join(f"{i}. {PARAGRAPH}".strip() for i in range(12))
    pieces = split_body(text, budget=800, overlap=100)
    for prev, cur in zip(pieces, pieces[1:], strict=False):
        assert cur.start < prev.end  # overlapping offsets
        assert text[cur.start : prev.end] in prev.text
    assert all(len(p.text) <= 800 for p in pieces)


def test_small_table_is_never_cut():
    table = "| A | B |\n|---|---|\n" + "".join(f"| r{i} | v{i} |\n" for i in range(10))
    text = PARAGRAPH + "\n\n" + table + "\n" + PARAGRAPH
    pieces = split_body(text, budget=len(table) + 20)
    holding = [p for p in pieces if "| A | B |" in p.text]
    assert len(holding) == 1 and table.strip() in holding[0].text


def test_oversized_table_splits_by_rows_with_header():
    header = "| Test | Value |\n|------|-------|\n"
    rows = [f"| row {i:02d} | value {i:02d} |" for i in range(20)]
    text = header + "\n".join(rows) + "\n"
    pieces = split_body(text, budget=200)
    assert len(pieces) > 1
    assert all(p.text.startswith(header) for p in pieces)
    assert all(len(p.text) <= 200 for p in pieces)
    assert [p.repeated_header for p in pieces] == [False] + [True] * (len(pieces) - 1)
    seen = [line for p in pieces for line in p.text.splitlines()[2:]]
    assert seen == rows


def test_oversized_code_block_keeps_fence_and_language():
    code = "\n\n".join(f"def f{i}():\n    return {i}" for i in range(20))
    text = f"```python\n{code}\n```\n"
    pieces = split_body(text, budget=150)
    assert len(pieces) > 1
    for p in pieces:
        assert p.text.startswith("```python\n") and p.text.endswith("\n```")
        assert len(p.text) <= 150


def test_cuts_prefer_words_and_hard_cut_is_last_resort():
    words = " ".join(f"word{i:03d}" for i in range(200))
    pieces = split_body(words, budget=300)
    for p in pieces:
        assert p.end == len(words) or words[p.end] == " "  # never mid-word
    unbroken = "x" * 1000
    hard = split_body(unbroken, budget=300)
    assert "".join(p.text for p in hard) == unbroken
    assert all(len(p.text) <= 300 for p in hard)


def test_find_blocks_partitions_text():
    text = "intro\n\n| a |\n|---|\n| 1 |\n\n```py\nx = 1\n```\nend\n"
    blocks = find_blocks(text)
    assert [b.kind for b in blocks] == ["text", "table", "text", "code", "text"]
    assert blocks[0].start == 0 and blocks[-1].end == len(text)
    assert all(a.end == b.start for a, b in zip(blocks, blocks[1:], strict=False))
