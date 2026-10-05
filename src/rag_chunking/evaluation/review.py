"""review.csv: automatic labels, suggested labels and the reviewer's final labels.

Three sources, kept in separate columns so it is always clear who decided what:
- *_auto: the automatic check against the frozen answer key (reproducible).
- *_suggested: a second, by-hand reading, loaded from results/review_suggestions.json.
- retrieval_label_final / answer_label: the human reviewer's decision, typed into the CSV.

Re-running refreshes the first two but never touches what the reviewer typed.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

from rag_chunking.evaluation.answer_key import AnswerKey
from rag_chunking.evaluation.autocheck import check_answer, check_retrieval

MANUAL_COLUMNS = ("retrieval_label_final", "answer_label", "notes")
SUGGESTION_COLUMNS = ("retrieval_label_suggested", "answer_label_suggested", "suggestion_reason")
SUGGESTIONS_FILE = "review_suggestions.json"
COLUMNS = (
    "question_id",
    "question",
    "context_ids",
    "context_tokens",
    "retrieval_auto",
    "facts_in_context",
    "answer",
    "answer_auto",
    "facts_in_answer",
    *SUGGESTION_COLUMNS,
    *MANUAL_COLUMNS,
)
VALID_LABELS = {"", "yes", "partial", "no", "N/A"}


def _load(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def read_review(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = {row["question_id"]: row for row in csv.DictReader(fh)}
    for qid, row in rows.items():
        for col in ("retrieval_label_final", "answer_label"):
            if row.get(col, "").strip() not in VALID_LABELS:
                raise ValueError(f"{path}: {qid} {col} = {row[col]!r}; use yes/partial/no")
    return rows


def answer_hash(answer: str) -> str:
    return hashlib.sha256(answer.encode("utf-8")).hexdigest()


def load_suggestions(path: Path) -> dict[str, dict[str, dict[str, str]]]:
    """{run_id: {question_id: {retrieval, answer, reason, context_ids, answer_sha256}}}.

    context_ids / answer_sha256 record exactly what was judged, so a suggestion is never
    applied to a context or answer it has not seen (e.g. after a re-generation).
    """
    data = _load(path) or {}
    for run_id, questions in data.items():
        for qid, s in questions.items():
            for field in ("retrieval", "answer"):
                if s.get(field, "") not in VALID_LABELS:
                    raise ValueError(f"{path}: {run_id} {qid} {field} = {s[field]!r}")
            if not s.get("reason"):
                raise ValueError(f"{path}: {run_id} {qid} has no reason")
            if s.get("retrieval") and not s.get("context_ids"):
                raise ValueError(f"{path}: {run_id} {qid} does not record the judged context")
            if s.get("answer") and not s.get("answer_sha256"):
                raise ValueError(f"{path}: {run_id} {qid} does not record the judged answer")
    return data


def build_rows(key: AnswerKey, run_dir: Path) -> list[dict[str, str]]:
    chunks = _load(run_dir / "chunks.json")
    retrievals = _load(run_dir / "retrievals.json")
    if chunks is None or retrievals is None:
        raise FileNotFoundError(f"{run_dir}: run the experiment first (chunks/retrievals missing)")
    answers = _load(run_dir / "answers.json")
    bodies = {c["id"]: c["body"] for c in chunks["chunks"]}
    by_question = {q["id"]: q for q in retrievals["questions"]}
    answer_by_question = {a["id"]: a["answer"] for a in answers["answers"]} if answers else {}

    rows = []
    for q in key.questions:
        r = by_question[q.id]
        retrieval = check_retrieval(q, [bodies[cid] for cid in r["context_ids"]])
        answer = answer_by_question.get(q.id)
        answered = check_answer(q, answer) if answer is not None else None
        rows.append(
            {
                "question_id": q.id,
                "question": q.question,
                "context_ids": " ".join(r["context_ids"]),
                "context_tokens": str(r["context_tokens"]),
                "retrieval_auto": retrieval.label,
                "facts_in_context": "; ".join(retrieval.matched),
                "answer": answer or "",
                "answer_auto": answered.label if answered else "",
                "facts_in_answer": "; ".join(answered.matched) if answered else "",
            }
        )
    return rows


def _apply_suggestion(row: dict[str, str], suggestion: dict[str, str]) -> None:
    same_context = suggestion.get("context_ids") == row["context_ids"]
    same_answer = suggestion.get("answer_sha256") == answer_hash(row["answer"])
    row["retrieval_label_suggested"] = suggestion.get("retrieval", "") if same_context else ""
    row["answer_label_suggested"] = suggestion.get("answer", "") if same_answer else ""
    stale = [
        what
        for what, used, ok in (
            ("context", suggestion.get("retrieval"), same_context),
            ("answer", suggestion.get("answer"), same_answer),
        )
        if used and not ok
    ]
    reason = suggestion.get("reason", "")
    if stale:
        reason = f"[not applied: the {' and '.join(stale)} changed since review] {reason}"
    row["suggestion_reason"] = reason


def write_review(key: AnswerKey, run_dir: Path) -> tuple[list[dict[str, str]], list[str]]:
    """Write review.csv. Returns (rows, warnings about manual labels on changed rows)."""
    path = run_dir / "review.csv"
    existing = read_review(path)
    suggestions = load_suggestions(run_dir.parent / SUGGESTIONS_FILE).get(run_dir.name, {})
    rows = build_rows(key, run_dir)
    warnings = []
    for row in rows:
        _apply_suggestion(row, suggestions.get(row["question_id"], {}))
        old = existing.get(row["question_id"], {})
        for col in MANUAL_COLUMNS:
            row[col] = old.get(col, "")
        # Manual labels are never discarded, but a label typed for a different context
        # or answer must not pass silently.
        if old and (old.get("retrieval_label_final") or old.get("answer_label")):
            changed = [c for c in ("context_ids", "answer") if old.get(c, "") != row[c]]
            if changed:
                warnings.append(
                    f"{run_dir.name} {row['question_id']}: {' and '.join(changed)} changed "
                    "since the manual label was entered; please re-check it"
                )
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows, warnings
