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
from rag_chunking.evaluation.review import answer_hash, read_review, write_review

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
    rows, _ = write_review(KEY, run_dir)
    m = run_metrics(KEY, run_dir, rows)
    # Q1 and Q8 retrieved (yes), the other 7 answerable questions not (no).
    assert m["retrieval_accuracy_auto"] == pytest.approx(100 * 2 / 9, abs=0.1)
    assert m["hit_at_1_top5"] == pytest.approx(2 / 9, abs=1e-3)
    assert m["answer_accuracy_auto"] == 10.0  # only the Q9 refusal is right
    # No human labels yet: the auto labels stand in, and the result is flagged provisional.
    assert m["answer_accuracy"] == 10.0
    assert m["answer_label_sources"] == {"final": 0, "suggested": 0, "auto": 10}
    assert m["provisional"] is True


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

    rows_again, warnings = write_review(KEY, run_dir)  # refresh
    assert warnings == []  # nothing changed under the manual labels
    kept = read_review(path)
    assert kept["Q2"]["retrieval_label_final"] == "yes"
    assert kept["Q2"]["notes"] == "manual, with a comma, and a\nnewline"
    assert kept["Q4"]["answer_label"] == "partial"
    m = run_metrics(KEY, run_dir, rows_again)
    assert m["answer_accuracy"] == 5.0  # all 10 labelled: one partial
    assert m["answer_label_sources"]["final"] == 10
    assert m["retrieval_label_sources"] == {"final": 1, "suggested": 0, "auto": 8}
    assert m["provisional"] is True  # 8 retrieval labels are still automatic


def test_suggestions_fill_their_own_columns_and_final_labels_win(tmp_path):
    run_dir = tmp_path / "dummy"
    _fake_run(run_dir)
    no_idea = answer_hash("no idea")
    suggestions = {
        "dummy": {
            "Q1": {
                "retrieval": "partial",
                "answer": "no",
                "reason": "header missing",
                "context_ids": "c-Q1",
                "answer_sha256": no_idea,
            },
            "Q8": {
                "retrieval": "yes",
                "answer": "yes",
                "reason": "fine",
                "context_ids": "c-Q8",
                "answer_sha256": no_idea,
            },
        }
    }
    (tmp_path / "review_suggestions.json").write_text(json.dumps(suggestions), encoding="utf-8")
    rows = {r["question_id"]: r for r in write_review(KEY, run_dir)[0]}
    assert rows["Q1"]["retrieval_label_suggested"] == "partial"
    assert rows["Q1"]["retrieval_label_final"] == ""  # suggestions never fill final columns
    assert rows["Q1"]["suggestion_reason"] == "header missing"
    m = run_metrics(KEY, run_dir, list(rows.values()))
    assert m["per_question"]["Q1"]["retrieval"] == "partial"  # suggestion beats auto (yes)
    assert m["retrieval_label_sources"] == {"final": 0, "suggested": 2, "auto": 7}

    path = run_dir / "review.csv"
    text = path.read_text(encoding="utf-8").replace("header missing,,,", "header missing,yes,,", 1)
    path.write_text(text, encoding="utf-8")
    rows, _ = write_review(KEY, run_dir)
    m = run_metrics(KEY, run_dir, rows)
    assert m["per_question"]["Q1"]["retrieval"] == "yes"  # the reviewer's label wins


def test_suggestion_without_reason_is_rejected(tmp_path):
    run_dir = tmp_path / "dummy"
    _fake_run(run_dir)
    bad = {"dummy": {"Q1": {"retrieval": "yes", "answer": "", "reason": "", "context_ids": "x"}}}
    (tmp_path / "review_suggestions.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="no reason"):
        write_review(KEY, run_dir)
    unbound = {"dummy": {"Q1": {"retrieval": "yes", "answer": "", "reason": "ok"}}}
    (tmp_path / "review_suggestions.json").write_text(json.dumps(unbound), encoding="utf-8")
    with pytest.raises(ValueError, match="judged context"):
        write_review(KEY, run_dir)


def _set_answer(run_dir: Path, qid: str, text: str) -> None:
    path = run_dir / "answers.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    for a in data["answers"]:
        if a["id"] == qid:
            a["answer"] = text
    path.write_text(json.dumps(data), encoding="utf-8")


def test_stale_suggestion_is_not_applied_after_regeneration(tmp_path):
    run_dir = tmp_path / "dummy"
    _fake_run(run_dir)
    suggestions = {
        "dummy": {
            "Q8": {
                "retrieval": "partial",
                "answer": "no",
                "reason": "judged an older answer",
                "context_ids": "c-Q8",
                "answer_sha256": answer_hash("no idea"),
            }
        }
    }
    (tmp_path / "review_suggestions.json").write_text(json.dumps(suggestions), encoding="utf-8")
    _set_answer(run_dir, "Q8", "Every 3-6 months.")  # a re-generated, different answer
    rows = {r["question_id"]: r for r in write_review(KEY, run_dir)[0]}
    assert rows["Q8"]["retrieval_label_suggested"] == "partial"  # same context: applies
    assert rows["Q8"]["answer_label_suggested"] == ""  # different answer: not applied
    assert rows["Q8"]["suggestion_reason"].startswith("[not applied: the answer changed")
    m = run_metrics(KEY, run_dir, list(rows.values()))
    assert m["per_question"]["Q8"]["answer"] == "yes"  # falls back to the automatic label


def test_manual_label_on_a_changed_answer_is_kept_but_flagged(tmp_path):
    run_dir = tmp_path / "dummy"
    _fake_run(run_dir)
    write_review(KEY, run_dir)
    path = run_dir / "review.csv"
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows[7]["answer_label"] = "no"  # Q8
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    _set_answer(run_dir, "Q8", "Every 3-6 months.")
    new_rows, warnings = write_review(KEY, run_dir)
    assert new_rows[7]["answer_label"] == "no"  # never discarded
    assert warnings == [
        "dummy Q8: answer changed since the manual label was entered; please re-check it"
    ]


def test_invalid_manual_label_is_rejected(tmp_path):
    run_dir = tmp_path / "dummy"
    _fake_run(run_dir)
    write_review(KEY, run_dir)
    path = run_dir / "review.csv"
    path.write_text(path.read_text(encoding="utf-8").replace(",,,\n", ",maybe,,\n", 1))
    with pytest.raises(ValueError, match="maybe"):
        read_review(path)
