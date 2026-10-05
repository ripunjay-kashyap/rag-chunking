"""review.csv: automatic labels next to empty columns for the manual review.

Re-running refreshes the automatic columns but never touches what the reviewer typed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from rag_chunking.evaluation.answer_key import AnswerKey
from rag_chunking.evaluation.autocheck import check_answer, check_retrieval

MANUAL_COLUMNS = ("retrieval_label_final", "answer_label", "notes")
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


def write_review(key: AnswerKey, run_dir: Path) -> list[dict[str, str]]:
    path = run_dir / "review.csv"
    existing = read_review(path)
    rows = build_rows(key, run_dir)
    for row in rows:
        old = existing.get(row["question_id"], {})
        for col in MANUAL_COLUMNS:
            row[col] = old.get(col, "")
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return rows
