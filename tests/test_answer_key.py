import json
from pathlib import Path

import pytest

from rag_chunking.evaluation.answer_key import (
    AnswerKeyError,
    Fact,
    load_answer_key,
    normalize_for_match,
    section_texts,
    validate_against_document,
)
from rag_chunking.normalize import normalize

KEY_PATH = Path("data/answer_key.json")
DOC = normalize(Path("data/diabetes_reference_document.md").read_text(encoding="utf-8"))


def _write(tmp_path: Path, raw: dict) -> Path:
    path = tmp_path / "key.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path


def test_real_key_matches_document():
    key = load_answer_key(KEY_PATH)
    assert [q.id for q in key.questions] == [f"Q{i}" for i in range(1, 11)]
    assert len(key.answerable) == 9
    validate_against_document(key, DOC)


def test_dash_and_case_insensitive_match():
    fact = Fact(name="range", accept=("5.7%–6.4%",), section="x")
    assert fact.matches("An HbA1c of 5.7%-6.4% indicates prediabetes")
    assert normalize_for_match("A  \n B—C") == "a b-c"


@pytest.mark.parametrize(
    "text",
    ["every 3-6 months", "every 3–6 months", "every 3 - 6 months", "every 3 to 6 months"],
)
def test_numeric_range_forms_are_equivalent(text):
    assert Fact(name="interval", accept=("3-6 months",), section="x").matches(text)


def test_range_rule_leaves_words_and_lone_numbers_alone():
    assert normalize_for_match("5.7% to 6.4%") == "5.7%-6.4%"
    assert normalize_for_match("Type 2 to adults") == "type 2 to adults"
    assert not Fact(name="interval", accept=("3-6 months",), section="x").matches("6 months")


def test_section_texts_stop_at_any_heading():
    sections = section_texts(DOC)
    assert "Diabetic retinopathy" in sections["6.1 Microvascular Complications"]
    assert "Coronary" not in sections["6.1 Microvascular Complications"]


def test_string_outside_its_section_is_rejected(tmp_path):
    raw = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    # 'microalbumin' is in the document (§7) but not in §6.1, which Q6 cites.
    raw["questions"][5]["required_facts"][0]["accept"] = ["microalbumin"]
    with pytest.raises(AnswerKeyError, match="Q6: 'microalbumin'"):
        validate_against_document(load_answer_key(_write(tmp_path, raw)), DOC)


def test_unanswerable_with_facts_is_rejected(tmp_path):
    raw = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    raw["questions"][8]["required_facts"] = raw["questions"][7]["required_facts"]
    with pytest.raises(AnswerKeyError, match="Q9"):
        load_answer_key(_write(tmp_path, raw))


def test_min_facts_out_of_range_is_rejected(tmp_path):
    raw = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    raw["questions"][0]["min_facts_for_yes"] = 3
    with pytest.raises(AnswerKeyError, match="min_facts_for_yes"):
        load_answer_key(_write(tmp_path, raw))


def test_source_section_without_facts_is_rejected(tmp_path):
    raw = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    raw["questions"][0]["source_sections"].append("3. Symptoms")
    with pytest.raises(AnswerKeyError, match="has no facts"):
        load_answer_key(_write(tmp_path, raw))
