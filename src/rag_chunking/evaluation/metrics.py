"""Per-run numbers for the assignment's results table."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rag_chunking.evaluation.answer_key import AnswerKey
from rag_chunking.evaluation.autocheck import SCORES, is_relevant


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def _accuracy(labels: list[str]) -> float | None:
    """Percentage, yes = 1, partial = 0.5, no = 0. None if any label is missing."""
    if not labels or any(label not in SCORES for label in labels):
        return None
    return round(100 * sum(SCORES[label] for label in labels) / len(labels), 1)


def run_metrics(key: AnswerKey, run_dir: Path, rows: list[dict[str, str]]) -> dict[str, Any]:
    chunks = json.loads((run_dir / "chunks.json").read_text(encoding="utf-8"))
    retrievals = json.loads((run_dir / "retrievals.json").read_text(encoding="utf-8"))
    bodies = {c["id"]: c["body"] for c in chunks["chunks"]}
    hits = {q["id"]: q["hits"] for q in retrievals["questions"]}
    row_by_id = {r["question_id"]: r for r in rows}
    answerable = key.answerable

    hit_at_1, reciprocal_ranks = [], []
    for q in answerable:
        ranks = [h["rank"] for h in hits[q.id] if is_relevant(q, bodies[h["chunk_id"]])]
        hit_at_1.append(1.0 if ranks and ranks[0] == 1 else 0.0)
        reciprocal_ranks.append(1 / ranks[0] if ranks else 0.0)

    # Manual labels win; the automatic label stands in until the reviewer fills one in.
    retrieval_final = [
        row_by_id[q.id]["retrieval_label_final"] or row_by_id[q.id]["retrieval_auto"]
        for q in answerable
    ]
    retrieval_auto = [row_by_id[q.id]["retrieval_auto"] for q in answerable]
    has_answers = any(r["answer"] for r in rows)
    answer_auto = [r["answer_auto"] for r in rows]
    manual_answers = [r["answer_label"] for r in rows]
    return {
        "run_id": retrievals["run_id"],
        "chunks": chunks["stats"],
        "mean_context_tokens": retrievals["mean_context_tokens"],
        "retrieval_accuracy_auto": _accuracy(retrieval_auto),
        "retrieval_accuracy": _accuracy(retrieval_final),
        "retrieval_manually_reviewed": sum(
            bool(row_by_id[q.id]["retrieval_label_final"]) for q in answerable
        ),
        f"hit_at_1_top{retrievals['retrieve_n']}": _mean(hit_at_1),
        f"mrr_top{retrievals['retrieve_n']}": _mean(reciprocal_ranks),
        "answer_accuracy_auto": _accuracy(answer_auto) if has_answers else None,
        # Only reported once every answer has a manual label.
        "answer_accuracy": _accuracy(manual_answers) if has_answers else None,
        "per_question": {
            r["question_id"]: {
                "retrieval": r["retrieval_label_final"] or r["retrieval_auto"],
                "answer": r["answer_label"] or r["answer_auto"] or None,
            }
            for r in rows
        },
    }
