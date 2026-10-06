"""`demo`: a one-screen walkthrough of the results, for presenting the project.

Read-only: it reads results/ and the config, makes no API calls and writes no files, so
it can be run live any number of times.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from rag_chunking.config import Config
from rag_chunking.evaluation.report import ABLATIONS, HEADLINE

_LABEL = {"yes": "yes", "partial": "partial", "no": "NO", "N/A": "n/a", "": "-"}


def _load(results_dir: Path, run_id: str) -> dict[str, Any]:
    run_dir = results_dir / run_id

    def read(name: str) -> dict[str, Any]:
        return json.loads((run_dir / name).read_text(encoding="utf-8"))

    with (run_dir / "review.csv").open(encoding="utf-8", newline="") as fh:
        rows = {r["question_id"]: r for r in csv.DictReader(fh)}
    retrievals = read("retrievals.json")
    return {
        "metrics": read("metrics.json"),
        "chunks": {c["id"]: c for c in read("chunks.json")["chunks"]},
        "questions": {q["id"]: q for q in retrievals["questions"]},
        "rows": rows,
    }


def _rule(title: str) -> str:
    return f"\n── {title} " + "─" * max(0, 78 - len(title))


def _snippet(text: str, limit: int = 70) -> str:
    flat = re.sub(r"\s+", " ", text).strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _pct(value: float | None) -> str:
    return "  —" if value is None else f"{value:5.1f}%"


def render(config: Config) -> str:
    results = config.settings.results_dir
    runs = {run_id: _load(results, run_id) for run_id in config.runs}
    a_id, b_id = HEADLINE
    a, b = runs[a_id], runs[b_id]
    s = config.settings
    out: list[str] = []

    out.append("RAG chunking comparison: fixed-size (A) vs structure-aware (B)")
    out.append(_rule("Setup"))
    out.append(f"Document     {s.document}  ({len(a['rows'])} fixed questions, Q9 unanswerable)")
    out.append(f"Embedding    {config.embedding.model}, cosine similarity, top-{s.k} to the LLM")
    out.append(f"Generation   {config.llm.model}, temperature {config.llm.temperature}")
    out.append(f"Headline     {a_id} (blind 500-char slices) vs {b_id} (one chunk per section)")

    out.append(_rule("1. Chunks"))
    out.append(f"{'Run':<17}{'Chunks':>7}{'Avg tok':>9}{'Min-max':>10}{'Context tok/q':>15}")
    for run_id in (a_id, b_id):
        m = runs[run_id]["metrics"]
        stats = m["chunks"]
        out.append(
            f"{run_id:<17}{stats['chunks']:>7}{stats['mean_tokens']:>9}"
            f"{str(stats['min_tokens']) + '-' + str(stats['max_tokens']):>10}"
            f"{m['mean_context_tokens']:>15}"
        )

    out.append(_rule("2. Why it matters: Q2 'What HbA1c range indicates prediabetes?'"))
    for run_id in (a_id, b_id):
        top = runs[run_id]["questions"]["Q2"]["context_ids"][0]
        chunk = runs[run_id]["chunks"][top]
        heading = " > ".join(chunk["heading_path"][1:]) or "(none)"
        out.append(f"{run_id:<10} rank-1 {top}")
        out.append(f"{'':<10} heading: {heading}")
        out.append(f"{'':<10} starts:  “{_snippet(chunk['body'], 60)}”")
    out.append("A's top chunk starts mid-table: the values are there, the row and column")
    out.append("headers are not. B's chunk carries the whole table under its heading.")

    out.append(_rule("3. All runs"))
    n = s.retrieve_n
    out.append(f"{'Run':<17}{'Retrieval':>10}{'hit@1':>7}{'MRR':>7}{'Ctx tok':>9}{'Answer':>9}")
    for run_id, run in runs.items():
        m = run["metrics"]
        out.append(
            f"{run_id:<17}{_pct(m['retrieval_accuracy']):>10}{m[f'hit_at_1_top{n}']:>7.2f}"
            f"{m[f'mrr_top{n}']:>7.2f}{m['mean_context_tokens']:>9}"
            f"{_pct(m['answer_accuracy']):>9}"
        )
    out.append("Retrieval over 9 answerable questions; answers over all 10 (yes=1, partial=0.5).")

    out.append(_rule(f"4. Per question: {a_id} vs {b_id}  (retrieval / answer)"))
    for qid, row in a["rows"].items():
        ra, rb = row, b["rows"][qid]
        cell_a = f"{_LABEL[ra['retrieval_label_final']]}/{_LABEL[ra['answer_label']]}"
        cell_b = f"{_LABEL[rb['retrieval_label_final']]}/{_LABEL[rb['answer_label']]}"
        out.append(f"{qid:<4} {cell_a:<16} {cell_b:<16} {_snippet(row['question'], 40)}")

    out.append(_rule("5. Ablations: one change each"))
    for title, ids, _ in ABLATIONS:
        cells = ", ".join(
            f"{i} {runs[i]['metrics']['retrieval_accuracy']:.1f}%" for i in ids if i in runs
        )
        out.append(f"{title:<38} {cells}")

    ma, mb = a["metrics"], b["metrics"]
    budget_a, budget_b = (runs[i]["metrics"] for i in ABLATIONS[0][1])
    out.append(_rule("Takeaways"))
    out.append(
        f"• B retrieves better: {mb['retrieval_accuracy']:.1f}% vs "
        f"{ma['retrieval_accuracy']:.1f}%, "
        f"and still wins at an equal budget ({budget_b['retrieval_accuracy']:.1f}% vs "
        f"{budget_a['retrieval_accuracy']:.1f}%)"
    )
    out.append(
        f"  with less text ({budget_b['mean_context_tokens']} vs "
        f"{budget_a['mean_context_tokens']} context tokens per question)."
    )
    out.append("• Blind cuts split tables and lists from their headers (A-300 Q1, A-800 Q1); the")
    out.append("  auto-check can't see that, which is why every row was reviewed by hand.")
    out.append(
        f"• The answer gap ({ma['answer_accuracy']:.0f}% vs {mb['answer_accuracy']:.0f}%) is "
        "noise: B's Q4 refusal is a generation"
    )
    out.append("  failure with the right chunk at rank 1 (see results/answer_stability.md).")
    out.append("\nDetails: results/comparison.md · labels: results/<run>/review.csv · README.md")
    return "\n".join(out)
