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


def resolve(row: dict[str, str], kind: str) -> tuple[str, str]:
    """(label, source): the reviewer's final label wins, then the suggestion, then auto."""
    final_col = "retrieval_label_final" if kind == "retrieval" else "answer_label"
    for source, col in (
        ("final", final_col),
        ("suggested", f"{kind}_label_suggested"),
        ("auto", f"{kind}_auto"),
    ):
        if row.get(col, "").strip():
            return row[col].strip(), source
    return "", "none"


def _sources(resolved: list[tuple[str, str]]) -> dict[str, int]:
    counts = {"final": 0, "suggested": 0, "auto": 0}
    for _, source in resolved:
        if source in counts:
            counts[source] += 1
    return counts


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

    retrieval = [resolve(row_by_id[q.id], "retrieval") for q in answerable]
    retrieval_auto = [row_by_id[q.id]["retrieval_auto"] for q in answerable]
    has_answers = any(r["answer"] for r in rows)
    answers = [resolve(r, "answer") for r in rows] if has_answers else []
    n = retrievals["retrieve_n"]
    return {
        "run_id": retrievals["run_id"],
        "chunks": chunks["stats"],
        "mean_context_tokens": retrievals["mean_context_tokens"],
        "retrieval_accuracy": _accuracy([label for label, _ in retrieval]),
        "retrieval_accuracy_auto": _accuracy(retrieval_auto),
        "retrieval_label_sources": _sources(retrieval),
        f"hit_at_1_top{n}": _mean(hit_at_1),
        f"mrr_top{n}": _mean(reciprocal_ranks),
        "answer_accuracy": _accuracy([label for label, _ in answers]) if has_answers else None,
        "answer_accuracy_auto": (
            _accuracy([r["answer_auto"] for r in rows]) if has_answers else None
        ),
        "answer_label_sources": _sources(answers) if has_answers else None,
        # Final only when every label came from the human reviewer.
        "provisional": any(source != "final" for _, source in retrieval + answers),
        "per_question": {
            r["question_id"]: {
                "retrieval": resolve(r, "retrieval")[0],
                "answer": resolve(r, "answer")[0] or None,
            }
            for r in rows
        },
    }
