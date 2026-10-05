import json
from pathlib import Path

import pytest

from rag_chunking.evaluation.answer_key import load_answer_key
from rag_chunking.evaluation.stability import compare

KEY = load_answer_key(Path("data/answer_key.json"))
REFUSAL = "The document does not contain this information."


def _write(root: Path, run_id: str, answers: dict[str, str], context: str = "c1") -> None:
    run_dir = root / run_id
    run_dir.mkdir(parents=True)
    rows = [{"id": q, "answer": a, "context_ids": [context]} for q, a in answers.items()]
    data = {"answers": rows}
    (run_dir / "answers.json").write_text(json.dumps(data), encoding="utf-8")


def _all(answer: str) -> dict[str, str]:
    return {q.id: answer for q in KEY.questions}


def test_counts_changed_texts_and_label_flips(tmp_path):
    first = _all("no idea") | {"Q8": "Every 3-6 months.", "Q9": REFUSAL}
    second = first | {"Q8": "Every 6 months.", "Q1": "no idea at all"}
    _write(tmp_path / "a", "run", first)
    _write(tmp_path / "b", "run", second)
    table = compare(KEY, tmp_path / "a", tmp_path / "b", ["run", "missing-run"])
    row = next(line for line in table.splitlines() if line.startswith("| `run`"))
    assert "| 20.0% | 10.0% | -10.0 | 2/10 | Q8: yes → no |" in row
    assert "missing-run" not in table  # runs without answers are skipped


def test_different_contexts_are_rejected(tmp_path):
    _write(tmp_path / "a", "run", _all("x"), context="c1")
    _write(tmp_path / "b", "run", _all("x"), context="c2")
    with pytest.raises(ValueError, match="contexts differ"):
        compare(KEY, tmp_path / "a", tmp_path / "b", ["run"])
