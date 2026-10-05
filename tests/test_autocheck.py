import csv
import json
from pathlib import Path

import pytest

from rag_chunking.evaluation.answer_key import load_answer_key
from rag_chunking.evaluation.autocheck import (
    check_answer,
    check_retrieval,
    is_refusal,
    is_relevant,
)
from rag_chunking.evaluation.metrics import run_metrics
from rag_chunking.evaluation.review import read_review, write_review

KEY = load_answer_key(Path("data/answer_key.json"))
Q = {q.id: q for q in KEY.questions}


def test_dash_variants_match():
    assert check_retrieval(Q["Q2"], ["prediabetes: 5.7%—6.4%"]).label == "yes"
    assert check_answer(Q["Q8"], "Every 3 to 6 months, depending on stability.").label == "yes"


def test_yes_partial_no():
    assert check_retrieval(Q["Q1"], ["6.5% or higher", "126 mg/dL or higher"]).label == "yes"
    assert check_retrieval(Q["Q1"], ["HbA1c of 6.5% or higher"]).label == "partial"
    assert check_retrieval(Q["Q1"], ["nothing relevant"]).label == "no"


def test_facts_do_not_match_across_a_chunk_boundary():
    assert check_retrieval(Q["Q8"], ["every 3-6", " months"]).label == "no"


def test_q5_needs_both_sections():
    only_51 = "5-7% weight, 150 minutes, whole grains, smoking, alcohol"
    assert check_retrieval(Q["Q5"], [only_51]).label == "partial"
    assert check_retrieval(Q["Q5"], [only_51, "inspect feet daily"]).label == "yes"


def test_unanswerable_question():
    assert check_retrieval(Q["Q9"], ["anything"]).label == "N/A"
    assert check_answer(Q["Q9"], "The documents do not contain this information.").label == "yes"
    assert check_answer(Q["Q9"], "Aim for 80-130 mg/dL before meals.").label == "no"
    hedged = "The document does not contain this information, but 70–130 mg/dL is typical."
    assert check_answer(Q["Q9"], hedged).label == "no"


def test_refusal_to_answerable_question_is_no():
    assert is_refusal("The document does not contain this information.")
    assert check_answer(Q["Q4"], "The document does not contain this information.").label == "no"


def test_eight_month_trap_fails_q8():
    assert check_answer(Q["Q8"], "Every 6 months for stable patients.").label == "no"


def test_relevance_for_ranking():
    assert is_relevant(Q["Q6"], "Diabetic retinopathy damages the retina")
    assert not is_relevant(Q["Q6"], "Annual foot exam")


def _fake_run(run_dir: Path) -> None:
    run_dir.mkdir(parents=True)
    body = {"Q1": "6.5% or higher and 126 mg/dL", "Q8": "every 3-6 months"}
    chunks = [{"id": f"c-{qid}", "body": text} for qid, text in body.items()]
    chunks.append({"id": "c-other", "body": "unrelated"})
    (run_dir / "chunks.json").write_text(
        json.dumps({"stats": {"chunks": 3}, "chunks": chunks}), encoding="utf-8"
    )
    questions = []
    for q in KEY.questions:
        top = f"c-{q.id}" if q.id in body else "c-other"
        others = [c["id"] for c in chunks if c["id"] != top]
        ids = [top, *others]
        questions.append(
            {
                "id": q.id,
                "context_ids": ids[:1],
                "context_tokens": 10,
                "hits": [{"rank": i + 1, "chunk_id": cid} for i, cid in enumerate(ids)],
            }
        )
    (run_dir / "retrievals.json").write_text(
        json.dumps(
            {"run_id": "dummy", "retrieve_n": 5, "mean_context_tokens": 10, "questions": questions}
        ),
        encoding="utf-8",
    )
    answers = [{"id": "Q9", "answer": "The document does not contain this information."}]
    answers += [{"id": q.id, "answer": "no idea"} for q in KEY.questions if q.id != "Q9"]
    (run_dir / "answers.json").write_text(json.dumps({"answers": answers}), encoding="utf-8")


def test_metrics_end_to_end_on_dummy_run(tmp_path):
    run_dir = tmp_path / "dummy"
    _fake_run(run_dir)
    rows = write_review(KEY, run_dir)
    m = run_metrics(KEY, run_dir, rows)
    # Q1 and Q8 retrieved (yes), the other 7 answerable questions not (no).
    assert m["retrieval_accuracy_auto"] == pytest.approx(100 * 2 / 9, abs=0.1)
    assert m["hit_at_1_top5"] == pytest.approx(2 / 9, abs=1e-3)
    assert m["answer_accuracy_auto"] == 10.0  # only the Q9 refusal is right
    assert m["answer_accuracy"] is None  # no manual labels yet


def test_review_round_trip_keeps_manual_labels(tmp_path):
    run_dir = tmp_path / "dummy"
    _fake_run(run_dir)
    write_review(KEY, run_dir)
    path = run_dir / "review.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["answer_label"] = "partial" if row["question_id"] == "Q4" else "no"
    rows[1]["retrieval_label_final"] = "yes"
    rows[1]["notes"] = "manual, with a comma, and a\nnewline"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    rows_again = write_review(KEY, run_dir)  # refresh
    kept = read_review(path)
    assert kept["Q2"]["retrieval_label_final"] == "yes"
    assert kept["Q2"]["notes"] == "manual, with a comma, and a\nnewline"
    assert kept["Q4"]["answer_label"] == "partial"
    m = run_metrics(KEY, run_dir, rows_again)
    assert m["answer_accuracy"] == 5.0  # all 10 labelled: one partial
    assert m["retrieval_manually_reviewed"] == 1


def test_invalid_manual_label_is_rejected(tmp_path):
    run_dir = tmp_path / "dummy"
    _fake_run(run_dir)
    write_review(KEY, run_dir)
    path = run_dir / "review.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",,,\n", ",maybe,,\n", 1))
    with pytest.raises(ValueError, match="maybe"):
        read_review(path)
