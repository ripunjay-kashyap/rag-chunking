import re
import shutil
from dataclasses import replace
from pathlib import Path

from rag_chunking.config import load_config
from rag_chunking.evaluation.answer_key import load_answer_key
from rag_chunking.evaluation.report import (
    README_END,
    README_START,
    _load_run,
    build_comparison,
    readme_block,
    update_readme,
    write_report,
)

CONFIG = load_config()
KEY = load_answer_key(Path("data/answer_key.json"))
README = Path("README.md")


def _runs(results_dir: Path = Path("results")):
    return {run_id: _load_run(results_dir, run_id) for run_id in CONFIG.runs}


def test_comparison_has_every_section_and_question():
    text = build_comparison(KEY, _runs())
    for heading in (
        "## Summary stats (all runs)",
        "## Where chunk boundaries fall",
        "## Headline: `A-500` vs `B-min100`, question by question",
        "## Ablation: Equal context budget",
        "## Ablation: Heading prefix",
        "## Ablation: Merge threshold",
        "## Ablation: Fixed size sensitivity",
        "## Answers and review notes",
    ):
        assert heading in text
    for q in KEY.questions:
        assert f"| {q.id} |" in text
        assert f"### {q.id}. {q.question}" in text
    for run_id in CONFIG.runs:
        assert f"| `{run_id}` |" in text


def test_report_is_deterministic(tmp_path):
    shutil.copytree("results", tmp_path / "results")
    config = replace(CONFIG, settings=replace(CONFIG.settings, results_dir=tmp_path / "results"))
    first, _ = write_report(config, KEY)
    content = first.read_bytes()
    second, _ = write_report(config, KEY)
    assert second.read_bytes() == content


def test_update_readme_replaces_only_the_marked_block(tmp_path):
    readme = tmp_path / "README.md"
    readme.write_text(f"intro\n{README_START}\nold\n{README_END}\noutro\n", encoding="utf-8")
    block = f"{README_START}\nnew $1 \\1\n{README_END}"  # regex specials stay literal
    assert update_readme(readme, block)
    assert readme.read_text(encoding="utf-8") == f"intro\n{block}\noutro\n"
    assert update_readme(readme, block)  # idempotent
    assert readme.read_text(encoding="utf-8") == f"intro\n{block}\noutro\n"

    plain = tmp_path / "plain.md"
    plain.write_text("no markers\n", encoding="utf-8")
    assert not update_readme(plain, block)
    assert plain.read_text(encoding="utf-8") == "no markers\n"


def test_readme_results_block_is_up_to_date():
    text = README.read_text(encoding="utf-8")
    current = text[text.index(README_START) : text.index(README_END) + len(README_END)]
    assert current == readme_block(KEY, _runs()), "run `python -m rag_chunking report`"


def test_readme_has_the_briefs_required_parts():
    text = README.read_text(encoding="utf-8")
    block = text[text.index(README_START) : text.index(README_END)]
    # Summary stats: number of chunks, average chunk size, retrieval accuracy per strategy.
    assert "| Chunks |" in block and "Avg size (tok)" in block and "Retrieval acc." in block
    # Question-by-question table with retrieved chunks, retrieval label and answer label.
    assert "chunks (rank order) | Retrieval | Answer" in block
    assert all(f"| {q.id} |" in block for q in KEY.questions)


def test_readme_analysis_is_200_to_400_words():
    text = README.read_text(encoding="utf-8")
    analysis = re.search(r"<!-- analysis:start -->(.*?)<!-- analysis:end -->", text, re.S)
    assert analysis, "analysis markers missing"
    words = re.findall(r"[A-Za-z0-9§%][^\s]*", analysis.group(1))
    assert 200 <= len(words) <= 400, len(words)


def test_demo_renders_from_committed_results_without_writing():
    from rag_chunking.demo import render

    results = Path("results")
    before = {p: p.stat().st_mtime_ns for p in results.rglob("*") if p.is_file()}
    text = render(load_config(Path("configs/experiments.toml")))
    after = {p: p.stat().st_mtime_ns for p in results.rglob("*") if p.is_file()}
    assert before == after
    assert "Takeaways" in text and "A-500" in text and "B-min100" in text
