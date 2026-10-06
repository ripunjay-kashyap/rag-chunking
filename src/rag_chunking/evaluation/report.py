"""Cross-run tables: results/comparison.md, plus the results block inside README.md.

Everything is read from the files under results/, so the report never calls an API and
re-running it after the reviewer fills in labels updates every table.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from rag_chunking.config import Config
from rag_chunking.evaluation.answer_key import AnswerKey

HEADLINE = ("A-500", "B-min100")
ABLATIONS = (
    (
        "Equal context budget (≤ 400 tokens)",
        ("A-500-budget", "B-min100-budget"),
        "Removes the volume confounder: at top-3, B passes ~44% more text than A.",
    ),
    (
        "Heading prefix",
        ("B-min100", "B-noprefix"),
        "Same boundaries, prefix on vs off.",
    ),
    (
        "Merge threshold",
        ("B-min40", "B-min100"),
        "B-min40 keeps §5.3, §6.1 and §6.2 as separate chunks.",
    ),
    (
        "Fixed size sensitivity",
        ("A-300", "A-500", "A-800"),
        "Same blind method, different cut positions.",
    ),
)
README_START = "<!-- results:start -->"
README_END = "<!-- results:end -->"
_SHORT = {"yes": "yes", "partial": "partial", "no": "**no**", "N/A": "N/A", "": "—"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_run(results_dir: Path, run_id: str) -> dict[str, Any]:
    run_dir = results_dir / run_id
    with (run_dir / "review.csv").open(encoding="utf-8", newline="") as fh:
        rows = {r["question_id"]: r for r in csv.DictReader(fh)}
    return {
        "id": run_id,
        "metrics": _load_json(run_dir / "metrics.json"),
        "chunks": _load_json(run_dir / "chunks.json"),
        "rows": rows,
    }


def _fmt(value: Any, suffix: str = "") -> str:
    return "—" if value is None else f"{value}{suffix}"


def _one_line(text: str, limit: int = 160) -> str:
    flat = re.sub(r"\s+", " ", text).strip().replace("|", "\\|")
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _count_sources(runs: list[dict[str, Any]], field: str) -> dict[str, int]:
    totals = {"final": 0, "suggested": 0, "auto": 0}
    for r in runs:
        for source, n in (r["metrics"][field] or {}).items():
            totals[source] += n
    return totals


def _sources_note(runs: list[dict[str, Any]]) -> str:
    ret = _count_sources(runs, "retrieval_label_sources")
    ans = _count_sources(runs, "answer_label_sources")
    if not ret["suggested"] + ret["auto"] + ans["suggested"] + ans["auto"]:
        return "All labels are the reviewer's final manual labels."
    # Ablation runs get a retrieval-only review by design, so their automatic answer
    # labels don't make the report provisional.
    if not ret["suggested"] + ret["auto"] + ans["suggested"]:
        return (
            f"**Labels.** All {ret['final']} retrieval labels and the {ans['final']} answer "
            f"labels of the two main runs are the reviewer's final manual labels. The "
            f"{ans['auto']} answer labels of the ablation runs are automatic (frozen-key "
            f"match): ablations were reviewed for retrieval only."
        )

    def describe(c: dict[str, int]) -> str:
        return f"{c['final']} final manual, {c['suggested']} suggested, {c['auto']} automatic"

    return (
        f"**Provisional labels.** Retrieval labels: {describe(ret)}. Answer labels: "
        f"{describe(ans)}. *Suggested* labels come from a by-hand reading of every "
        f"retrieved context and answer, each with a written reason "
        f"(`results/review_suggestions.json`); *automatic* labels come from matching the "
        f"frozen answer key. They are replaced by the reviewer's final labels once "
        f"`retrieval_label_final` / `answer_label` are filled in each `review.csv` and "
        f"`review --all` and `report` are re-run."
    )


def summary_table(runs: list[dict[str, Any]]) -> str:
    lines = [
        "| Run | Strategy | Chunks | Avg size (tok) | Min–max | Context tok/question "
        "| Retrieval acc. | hit@1 | MRR | Answer acc. |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        m, s = r["metrics"], r["metrics"]["chunks"]
        strategy = "A fixed" if r["chunks"]["strategy"] == "fixed" else "B structure"
        lines.append(
            f"| `{r['id']}` | {strategy} | {s['chunks']} | {s['mean_tokens']} "
            f"| {s['min_tokens']}–{s['max_tokens']} | {m['mean_context_tokens']} "
            f"| {_fmt(m['retrieval_accuracy'], '%')} | {_fmt(m['hit_at_1_top5'])} "
            f"| {_fmt(m['mrr_top5'])} | {_fmt(m['answer_accuracy'], '%')} |"
        )
    return "\n".join(lines)


def boundary_table(runs: list[dict[str, Any]]) -> str:
    lines = [
        "| Run | Boundaries | In a table row | In a list item | In a heading "
        "| Mid-sentence | Clean (line break) | Mid-word |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        c = r["chunks"]["cut_summary"]
        total = len(r["chunks"]["cuts"])
        lines.append(
            f"| `{r['id']}` | {total} | {c['table-row']} | {c['list-item']} | {c['heading']} "
            f"| {c['mid-sentence'] + c['sentence-end']} | {c['clean']} | {c['mid-word']} |"
        )
    return "\n".join(lines)


def _label(row: dict[str, str], kind: str) -> str:
    final = row["retrieval_label_final"] if kind == "retrieval" else row["answer_label"]
    label = final or row[f"{kind}_label_suggested"] or row[f"{kind}_auto"]
    return _SHORT.get(label, label)


def headline_table(key: AnswerKey, a: dict[str, Any], b: dict[str, Any]) -> str:
    lines = [
        f"| Q | `{a['id']}` chunks (rank order) | Retrieval | Answer "
        f"| `{b['id']}` chunks (rank order) | Retrieval | Answer |",
        "|---|---|---|---|---|---|---|",
    ]
    for q in key.questions:
        ra, rb = a["rows"][q.id], b["rows"][q.id]
        lines.append(
            f"| {q.id} | {ra['context_ids'].replace(' ', ', ')} | {_label(ra, 'retrieval')} "
            f"| {_label(ra, 'answer')} | {rb['context_ids'].replace(' ', ', ')} "
            f"| {_label(rb, 'retrieval')} | {_label(rb, 'answer')} |"
        )
    return "\n".join(lines)


def ablation_table(key: AnswerKey, runs: list[dict[str, Any]]) -> str:
    head = " | ".join(f"`{r['id']}`" for r in runs)
    lines = [f"| Q | {head} |", "|---|" + "---|" * len(runs)]
    for q in key.answerable:
        cells = " | ".join(
            f"{_label(r['rows'][q.id], 'retrieval')} ({r['rows'][q.id]['context_tokens']} tok)"
            for r in runs
        )
        lines.append(f"| {q.id} | {cells} |")
    acc = " | ".join(f"**{_fmt(r['metrics']['retrieval_accuracy'], '%')}**" for r in runs)
    lines.append(f"| Retrieval acc. | {acc} |")
    return "\n".join(lines)


def answers_appendix(key: AnswerKey, runs: list[dict[str, Any]]) -> str:
    out = []
    for q in key.questions:
        out.append(f"### {q.id}. {q.question}\n")
        out.append(f"*Expected:* {q.expected_answer}\n")
        for r in runs:
            row = r["rows"][q.id]
            reason = row["suggestion_reason"]
            out.append(
                f"- **`{r['id']}`** — retrieval {_label(row, 'retrieval')}, "
                f"answer {_label(row, 'answer')}. Chunks: {row['context_ids']}.  \n"
                f"  > {_one_line(row['answer'], 600)}"
            )
            if row["notes"]:
                out.append(f"  \n  *Reviewer:* {_one_line(row['notes'], 400)}")
            elif reason:
                out.append(f"  \n  *Review note:* {_one_line(reason, 600)}")
        out.append("")
    return "\n".join(out)


def readme_block(key: AnswerKey, runs_by_id: dict[str, dict[str, Any]]) -> str:
    a, b = runs_by_id[HEADLINE[0]], runs_by_id[HEADLINE[1]]
    return "\n\n".join(
        [
            README_START,
            "_Generated by `python -m rag_chunking report` from `results/`. Do not edit by hand._",
            _sources_note(list(runs_by_id.values())),
            "#### Summary stats",
            f"Headline comparison: `{a['id']}` vs `{b['id']}`. The other rows are the "
            "ablations described below. Token sizes are `ceil(chars / 4)` estimates of the "
            "embedded text (B includes its heading prefix).",
            summary_table(list(runs_by_id.values())),
            "#### Question by question",
            "Retrieval = do the top-3 chunks contain the correct information "
            "(yes / partial / no). Answer = is the generated answer correct. "
            "Q9 is not answered in the document, so its retrieval is N/A and the "
            "answer is scored on a correct refusal.",
            headline_table(key, a, b),
            README_END,
        ]
    )


def build_comparison(key: AnswerKey, runs_by_id: dict[str, dict[str, Any]]) -> str:
    runs = list(runs_by_id.values())
    a, b = runs_by_id[HEADLINE[0]], runs_by_id[HEADLINE[1]]
    parts = [
        "# Comparison of chunking strategies",
        "_Generated by `python -m rag_chunking report` from the files in `results/`._",
        _sources_note(runs),
        "Scoring: yes = 1, partial = 0.5, no = 0. Retrieval accuracy is over the 9 "
        "answerable questions (Q9 is N/A); answer accuracy is over all 10, with Q9 scored "
        "on a correct refusal. hit@1 / MRR use the top 5, where a chunk is relevant if it "
        "contains at least one required fact. Runs `A-300` and `A-800` are retrieval-only.",
        "## Summary stats (all runs)",
        summary_table(runs),
        "## Where chunk boundaries fall",
        "Every interior chunk start and end, classified by what it cuts through.",
        boundary_table(runs),
        f"## Headline: `{a['id']}` vs `{b['id']}`, question by question",
        headline_table(key, a, b),
    ]
    for title, ids, why in ABLATIONS:
        parts += [f"## Ablation: {title}", why, ablation_table(key, [runs_by_id[i] for i in ids])]
    parts += [
        f"## Answers and review notes (`{a['id']}`, `{b['id']}`)",
        answers_appendix(key, [a, b]),
    ]
    return "\n\n".join(parts).rstrip() + "\n"


def update_readme(readme: Path, block: str) -> bool:
    """Replace the marked results block. Returns False if the markers are missing."""
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + ".*?" + re.escape(README_END), re.DOTALL)
    if not pattern.search(text):
        return False
    readme.write_text(pattern.sub(lambda _: block, text), encoding="utf-8")
    return True


def write_report(config: Config, key: AnswerKey, readme: Path | None = None) -> tuple[Path, bool]:
    """Write comparison.md; refresh README's results block. Returns (path, readme_updated)."""
    results_dir = config.settings.results_dir
    runs_by_id = {run_id: _load_run(results_dir, run_id) for run_id in config.runs}
    out = results_dir / "comparison.md"
    out.write_text(build_comparison(key, runs_by_id), encoding="utf-8")
    updated = False
    if readme is not None and readme.is_file():
        updated = update_readme(readme, readme_block(key, runs_by_id))
    return out, updated
