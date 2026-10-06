# RAG Chunking Strategy Comparison

A small RAG pipeline over `data/diabetes_reference_document.md`, built twice with
different chunking:

- **Strategy A (fixed-size):** blind 500-character windows with 50 characters of overlap.
- **Strategy B (structure-aware):** one chunk per Markdown section, with a heading-path
  prefix, small-sibling merging and a recursive fallback for oversized sections.

Both run through the same embedding model, retriever, LLM, prompt and 10 test
questions. Each comparison changes exactly one thing.

**Short answer:** B retrieved better (100% vs 94.4% retrieval accuracy, hit@1 1.0 vs
0.889) and kept that lead with *less* text under an equal token budget. Its answers were
not measurably better. In the committed run B scored 80% vs A's 90%, but a second
generation of the identical prompts gave 90% vs 90%: at temperature 0, answer accuracy
moved by up to 10 points between runs, while retrieval reproduced exactly
([`results/answer_stability.md`](results/answer_stability.md)). Details, ablations and
the written analysis follow.

## Contents

1. [How to run](#1-how-to-run)
2. [The document](#2-the-document)
3. [Strategy A and Strategy B](#3-strategy-a-and-strategy-b)
4. [Pipeline](#4-pipeline)
5. [Evaluation method](#5-evaluation-method)
6. [Results](#6-results)
7. [Ablations](#7-ablations)
8. [Analysis](#8-analysis)
9. [Decision log](#9-decision-log)
10. [Limitations and production next steps](#10-limitations-and-production-next-steps)
11. [Repository layout](#11-repository-layout)

---

## 1. How to run

Requires Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                                   # install
cp .env.example .env                      # add GEMINI_API_KEY (Google AI Studio)
uv run pytest                             # full test suite, no API calls

uv run python -m rag_chunking list                    # the 8 configured runs
uv run python -m rag_chunking chunk --run B-min100    # inspect chunks (no API calls)
uv run python -m rag_chunking run --all               # chunk → embed → retrieve → generate → auto-check
uv run python -m rag_chunking review --all            # refresh review.csv + metrics.json
uv run python -m rag_chunking report                  # results/comparison.md + tables in this README
uv run python -m rag_chunking stability --other results/rerun   # answer noise between two generations
```

- **Caching:** every API call goes through a disk cache (`.cache/`), so a re-run with a
  warm cache makes zero calls. A cold `run --all` makes 7 embedding calls and 43
  generation calls. That fits the Gemini free tier, and the client throttles itself to the
  limits in the config.
- **Determinism:** temperature is 0, chunk IDs are stable and ties are broken by
  document order. Re-running produces byte-identical `chunks.json`, `retrievals.json`,
  `review.csv` and `metrics.json`. `manifest.json` is the run log (timestamp, git commit,
  API call counts), so it is the one file that changes.
- **Swapping the LLM is a config change.** Set `[generation] active = "groq"` (needs
  `GROQ_API_KEY`) or `"ollama"` (local, no key) in `configs/experiments.toml`. Both use
  the OpenAI-compatible client. The embedding model, dimensions, templates and rate
  limits are in the same file. No model ID appears in the code.

## 2. The document

7,535 characters (≈1,884 tokens), one `#` title, ten `##` sections and five `###`
subsections. It's clean Markdown:

- one 4-column table (§4, diagnostic thresholds)
- 46 bullet items across most sections
- an FAQ (§10) whose questions are bold text, not headings

The 13 sections with content are 77–213 tokens each, so a whole section fits in one
chunk with room to spare. Three sections (the title, §5 and §6) are pure parents with no
text of their own.

This shape suits structure-aware chunking: the headings already mark the topic
boundaries, and every test question maps to one or two sections. It also makes the test
less dramatic than it would be on a long, messy document. With only 11–17 chunks, a
top-3 retrieval covers a large share of the document whatever the chunking.

## 3. Strategy A and Strategy B

### Strategy A: blind fixed-size (`src/rag_chunking/chunking/fixed.py`)

`text[i : i + 500]` with a step of 450 (50 characters of overlap), as the brief suggests.
No separators, no whitespace snapping, no heading prefix, and the short tail chunk is kept
as is. LangChain's `CharacterTextSplitter` was **not** used because it splits on
`"\n\n"` by default, which already makes it partly structure-aware and would blur the
comparison. Chunk sizes are counted in characters, so the boundaries can't move when the
embedding model (and its tokenizer) changes.

### Strategy B: structure-aware with recursive fallback (`structure.py`, `recursive.py`)

The steps run in order: parse sections, merge the small ones, then split the oversized
ones.

1. **Parse** `#`/`##`/`###` headings, ignoring `#` lines inside fenced code. Each
   section gets its heading path, e.g. `Type 2 Diabetes… > 6. Complications > 6.1
   Microvascular Complications`.
2. **Empty parents** (title, §5, §6) produce no chunk. Their titles live on in their
   children's paths.
3. **Heading prefix:** each chunk's text is `path + "\n\n" + body`. A section like
   "These affect small blood vessels…" is ambiguous without its heading.
4. **Min rule (junk filter):** a section under `min_tokens` (100) of body text merges
   with an adjacent sibling under the same parent, preferring the previous one. A merged
   chunk takes the parent's path and keeps the sub-headings in its body. On this document,
   §5.3 merges into §5.2, and §6.1 with §6.2.
5. **Max rule (safety net):** a chunk over 512 tokens is split by a recursive fallback.
   - It cuts at `\n\n`, then `\n`, then a sentence end, then a space, then a hard cut,
     whichever comes first in that order near the target position.
   - The piece count is chosen up front, so pieces come out even with no tiny tail.
   - Tables and code blocks are atomic. An oversized table is split by rows with its
     header repeated, and an oversized code block re-opens its fence in each piece.
   - This never triggers on this document. It's unit-tested with smaller limits.

Semantic chunking (split where embedding similarity drops) was considered. It needs an
embedding call per sentence, depends on the model, and is hard to cite. When a document
already has good headings, the headings are a cheaper and more faithful topic signal.

| Run | Chunks | Avg tokens | Boundaries that cut a table / list / word |
|---|---:|---:|---|
| A-500 | 17 | 122.6 | 2 / 14 / 20 of 32 |
| B-min100 | 11 | 184.1 | 0 / 0 / 0 of 21 (all on section edges) |

The full boundary classification is in `results/comparison.md`. `uv run python -m
rag_chunking chunk --run A-500` prints every chunk with its first and last 60 characters.

## 4. Pipeline

- **Embeddings:** Gemini `gemini-embedding-2`, 768 dimensions, L2-normalized. This model
  has no `task_type`, so retrieval intent goes into the text using Google's templates:
  documents `title: none | text: …`, queries `task: search result | query: …`. The
  templates are identical for both strategies. Each text is sent as its own `Content`
  object, because sending a list of plain strings returns *one* aggregated vector
  (verified live).
- **Vector store:** a NumPy matrix with a dot product (cosine on unit vectors). There are
  11–28 vectors per run, so exact search is trivial, and FAISS or Chroma would only hide
  the mechanics.
- **Retrieval:** top-k with k = 3, with the top 5 logged for hit@1 and MRR. There is also
  a **token-budget mode** (≤ 400 tokens, chunks taken in rank order) to compare strategies
  at equal context volume.
- **Generation:** `gemini-3.5-flash-lite`, temperature 0, with one grounded prompt for
  every run: answer only from the sources, refuse with a fixed sentence, flag false
  premises, keep ranges as written, and cite sources. Sources are numbered `[1]…[n]`,
  **not** labelled with chunk IDs, because B's IDs (e.g. `B-4-diagnostic-criteria`) would
  leak heading information even into the no-prefix run. The planned `gemini-3.8-flash`
  allows only 20 free requests a day, so the lite model was used for all runs.
- **Generated answers:** produced for six runs. `A-300` and `A-800` are retrieval-only by
  design (a chunk-size ablation).

## 5. Evaluation method

### Answer key, frozen before any run (`data/answer_key.json`)

For each question the key records:

- the source sections
- the required facts, each with accepted strings that are validated to occur *in that
  section*
- how many facts are needed for "yes"
- a reference answer
- the traps

It was frozen before any retrieval ran, so labels can't drift toward a preferred
strategy. It has not changed since.

**Traps built into the brief's questions**
- **Q3** asks for "three first-line" drugs, but the document calls only Metformin
  first-line. A good answer flags the false premise.
- **Q5** needs two sections: §5.1 (lifestyle modifications) **and** §8 (self-management
  tips).
- **Q8** asks about "stable" patients. The document says "every 3-6 months, depending on
  how stable", so inventing "6 months" is wrong.
- **Q9** (glucose *target* range) is **not in the document**. It's excluded from
  retrieval accuracy (N/A), and the answer scores only for a correct refusal.

### Scoring

- **Labels:** yes = 1, partial = 0.5, no = 0.
- **Yes:** at least `min_facts_for_yes` facts are found, **and** every source section has
  at least one. **Partial:** at least one fact, but not yes. **No:** none.
- **Retrieval accuracy:** the rule applied to the top-3 chunks (facts matched *per
  chunk*, never across a boundary), over the 9 answerable questions.
- **Answer accuracy:** the rule applied to the generated answer, over all 10 questions.
  A refusal to an answerable question scores no.
- **hit@1 / MRR over the top 5:** a chunk is relevant if it holds at least one required
  fact.
- **Matching:** case-insensitive, with dashes and whitespace unified, and numeric ranges
  written one way ("3 to 6" → "3-6"). The range rule is content-blind, applies to chunks
  and answers alike, and was fixed before any run. Synonyms are deliberately not accepted
  and go to manual review instead.

### Three layers of labels (`results/<run>/review.csv`)

1. **Automatic:** matches the frozen key. It's reproducible, but it can't see a missing
   table header or a lure.
2. **Suggested:** a by-hand reading of every retrieved context and every answer, each with
   a written reason (`results/review_suggestions.json`). It disagrees with the automatic
   label once in 92 rows: A-800 Q1, where "126 mg/dL" was retrieved without its row label
   or column header. Each suggestion records the exact context and answer it judged
   (chunk IDs and a hash of the answer). If either changes, for example after a
   re-generation, the suggestion is not applied and the row falls back to the automatic
   label.
3. **Final:** the human reviewer's `retrieval_label_final` / `answer_label` columns. When
   they are filled in, they override everything above, and `report` regenerates every
   table.

The tables state which layer each number comes from.

## 6. Results

<!-- results:start -->

_Generated by `python -m rag_chunking report` from `results/`. Do not edit by hand._

**Labels.** All 72 retrieval labels and the 20 answer labels of the two main runs are the reviewer's final manual labels. The 40 answer labels of the ablation runs are automatic (frozen-key match): ablations were reviewed for retrieval only.

#### Summary stats

Headline comparison: `A-500` vs `B-min100`. The other rows are the ablations described below. Token sizes are `ceil(chars / 4)` estimates of the embedded text (B includes its heading prefix).

| Run | Strategy | Chunks | Avg size (tok) | Min–max | Context tok/question | Retrieval acc. | hit@1 | MRR | Answer acc. |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `A-500` | A fixed | 17 | 122.6 | 84–125 | 366.8 | 94.4% | 0.889 | 0.944 | 90.0% |
| `A-300` | A fixed | 28 | 74.5 | 62–75 | 225.0 | 83.3% | 0.889 | 0.911 | — |
| `A-800` | A fixed | 11 | 189.5 | 84–200 | 565.2 | 88.9% | 0.889 | 0.944 | — |
| `A-500-budget` | A fixed | 17 | 122.6 | 84–125 | 366.8 | 94.4% | 0.889 | 0.944 | 90.0% |
| `B-min100` | B structure | 11 | 184.1 | 144–289 | 529.0 | 100.0% | 1.0 | 1.0 | 80.0% |
| `B-min40` | B structure | 13 | 158.5 | 104–236 | 492.6 | 100.0% | 1.0 | 1.0 | 85.0% |
| `B-noprefix` | B structure | 11 | 162.5 | 118–269 | 462.2 | 94.4% | 1.0 | 1.0 | 90.0% |
| `B-min100-budget` | B structure | 11 | 184.1 | 144–289 | 333.5 | 100.0% | 1.0 | 1.0 | 85.0% |

#### Question by question

Retrieval = do the top-3 chunks contain the correct information (yes / partial / no). Answer = is the generated answer correct. Q9 is not answered in the document, so its retrieval is N/A and the answer is scored on a correct refusal.

| Q | `A-500` chunks (rank order) | Retrieval | Answer | `B-min100` chunks (rank order) | Retrieval | Answer |
|---|---|---|---|---|---|---|
| Q1 | A-004, A-005, A-000 | yes | yes | B-4-diagnostic-criteria, B-1-overview, B-7-monitoring-and-follow-up | yes | yes |
| Q2 | A-005, A-004, A-016 | yes | yes | B-4-diagnostic-criteria, B-7-monitoring-and-follow-up, B-10-frequently-asked-questions | yes | yes |
| Q3 | A-007, A-008, A-006 | yes | partial | B-5-treatment-approaches, B-5-1-lifestyle-modifications, B-10-frequently-asked-questions | yes | partial |
| Q4 | A-004, A-003, A-014 | yes | yes | B-3-symptoms, B-9-when-to-seek-emergency-care, B-4-diagnostic-criteria | yes | **no** |
| Q5 | A-006, A-016, A-007 | partial | partial | B-5-1-lifestyle-modifications, B-8-self-management-and-lifestyle-tips, B-10-frequently-asked-questions | yes | partial |
| Q6 | A-010, A-009, A-011 | yes | yes | B-6-complications, B-7-monitoring-and-follow-up, B-1-overview | yes | yes |
| Q7 | A-010, A-009, A-011 | yes | yes | B-6-complications, B-1-overview, B-7-monitoring-and-follow-up | yes | yes |
| Q8 | A-011, A-012, A-000 | yes | yes | B-7-monitoring-and-follow-up, B-4-diagnostic-criteria, B-8-self-management-and-lifestyle-tips | yes | yes |
| Q9 | A-005, A-000, A-015 | N/A | yes | B-4-diagnostic-criteria, B-8-self-management-and-lifestyle-tips, B-5-1-lifestyle-modifications | N/A | yes |
| Q10 | A-013, A-014, A-012 | yes | yes | B-9-when-to-seek-emergency-care, B-3-symptoms, B-8-self-management-and-lifestyle-tips | yes | yes |

<!-- results:end -->

Every chunk, ranking, score and answer is in `results/<run>/` (`chunks.json`,
`retrievals.json`, `answers.json`, `review.csv`, `metrics.json`, `manifest.json`). The
full report, with ablation tables and every answer, is
[`results/comparison.md`](results/comparison.md).

## 7. Ablations

Each row changes one thing relative to its partner (numbers in the summary table above):

| Comparison | What changes | Result |
|---|---|---|
| `A-500-budget` vs `B-min100-budget` | Equal context (≤ 400 tokens) | B 100% with 333.5 tokens per question vs A 94.4% with 366.8. B wins with *less* text, so the top-3 lead isn't explained by volume. |
| `B-min100` vs `B-noprefix` | Heading prefix | 100% → 94.4%. Without the prefix, §8 "Self-Management and **Lifestyle** Tips" drops from rank 2 to rank 4 for Q5. |
| `B-min40` vs `B-min100` | Merge threshold | Both 100%. Unmerged §6.1 / §6.2 give tighter context for Q6/Q7 (364–382 vs 509 tokens), with the same correctness. |
| `A-300` / `A-500` / `A-800` | Fixed size | 83.3% / 94.4% / 88.9%. Not monotonic: Q1 depends on where the cut lands in the §4 table (no / yes / partial). |

## 8. Analysis

<!-- analysis:start -->
**Structure-aware chunking (B) retrieved better; it did not answer better.** At top-3,
B-min100 found the right information for all nine answerable questions (100% vs 94.4%
for A-500) and ranked a relevant chunk first every time (hit@1 1.0 vs 0.889). The margin
is small because the document is short and cleanly structured: with 11–17 chunks, almost
any top-3 covers the answer.

Where B clearly won, the mechanism is visible. Q5 needs two sections (§5.1 and §8).
Every B run with heading prefixes retrieved both; every A run retrieved only §5.1, and
removing the prefix made B miss §8 too. Q1 shows the table risk: A-500 was right only
because its cut fell after the FPG and HbA1c rows. At 300 characters the values landed in
a chunk without the header (no); at 800 they lost their row label (partial). A's accuracy
across sizes (83/94/89%) is not monotonic, because it depends on where cuts land, not on
chunk size. A also ranked lure fragments first: for Q4 its top chunk was the tail of §3,
containing "hyperglycemia" but none of the symptoms.

Volume does not explain the gap. At top-3, B passes about 44% more text (529 vs 367
tokens), but under an equal 400-token budget it still scored 100% using less text (334
tokens).

Answers did not separate the strategies. In the committed run B-min100 scored 80%
against A's 90%, entirely from one Q4 refusal despite the full symptom list ranking
first. Regenerating the identical prompts from a clean clone changed 6–8 of 10 answer
texts per run and turned that refusal into a correct answer (90% vs 90%). So differences
of 5–10 points are run-to-run noise here, while retrieval reproduced exactly. Both
strategies handled the traps alike: the Q3 premise was flagged, the Q8 range kept, and
Q9 refused.

Trade-offs: A is 46 lines with one parameter and chunks in 0.12 ms. B is about 430 lines
of merge, fallback and atomic-table rules and takes 0.32 ms: negligible at runtime, but
real code to test and maintain. B's advantage also depends on good headings; on
unstructured text it falls back to recursive splitting. For this document B is worth it,
mainly for robustness, because A's good result here was partly luck.
<!-- analysis:end -->

## 9. Decision log

| Decision | Options considered | Choice | Why |
|---|---|---|---|
| Strategy A splitter | LangChain `CharacterTextSplitter`, own slicing | Own `text[i:i+500]` slicing | The LangChain splitter splits on `\n\n` first, so it isn't blind |
| A's size unit | Tokens, characters | Characters (500 / 50, as in the brief) | Boundaries independent of any tokenizer |
| Strategy B family | Recursive splitter, Markdown header splitter, semantic chunking | Header sections + own recursive fallback | Clean headings; deterministic, cheap, citable; semantic needs per-sentence embeddings |
| Heading context | None, title field, prefix in text | Path prefix in the embedded text | "These affect small blood vessels…" is ambiguous alone; ablated in `B-noprefix` |
| Small sections | Keep, drop, merge | Merge into an adjacent sibling, same parent only, min 100 (40 tested) | Thin chunks embed poorly; never mix topics across parents |
| Min rule unit | Prefixed text, body | Body tokens | Counting the prefix made min 100 = min 40 and tied boundaries to the prefix toggle |
| Oversized sections | Greedy fill, even split | Piece count up front, cut at the coarsest separator near each target | No tiny tail chunks |
| Tables / code | Split like text, atomic | Atomic; split by rows with the header repeated if oversized | A row without its header is meaningless (see A-300 Q1) |
| Empty parent headings | Own chunk, skip | Skip; kept in children's paths | Avoids near-empty junk chunks |
| FAQ (§10) | Split on bold questions, one chunk | One chunk | No test question targets it; listed as a limitation |
| Token counting | Model tokenizer, `ceil(chars/4)` | `ceil(chars/4)` | Model-independent; swapping models never moves boundaries |
| Normalization | Hand-edit the document, code | Code: line endings, trailing spaces, blank lines | Production pipelines can't hand-edit inputs |
| Embeddings | MiniLM, OpenAI, Gemini | `gemini-embedding-2`, 768 dims, behind an `Embedder` interface | One provider and key for embeddings and generation; swappable |
| Embedding intent | `task_type`, templates | Google's text templates, same for both strategies | The model has no `task_type`; `title: none` keeps B's heading in the text, not in a field favouring B |
| Vector store | FAISS, Chroma, NumPy | NumPy | 11–28 vectors; transparency over infrastructure |
| k | 1, 3, 5 | 3 (top 5 logged) | ≈ 20% of the document; hit@1 / MRR catch ranking differences |
| Volume confounder | Ignore, equal budget | Extra runs with a 400-token budget | B's chunks are bigger; separates boundaries from amount of text |
| LLM | Gemini 3.8 Flash, 3.5 Flash-Lite, Groq Llama | `gemini-3.5-flash-lite`, temperature 0 | 3.8 Flash's free tier is 20 requests/day; one provider; Groq/Ollama via config |
| Source labels in prompt | Chunk IDs, numbers | `[1]…[n]` | B's IDs name sections and would leak heading context |
| Prompt tuning | Tune after seeing answers, freeze | Frozen | Tuning on the test questions would bias the comparison; the Q4 refusal was reported instead (and turned out to be run-to-run noise) |
| Scoring | LLM-as-a-judge, RAGAS, key + manual | Frozen key → automatic → suggested → human | The brief asks for manual judgement; the key makes it reproducible |

## 10. Limitations and production next steps

- **Small test:** one short, clean document and 10 questions. A one-question difference
  is 11 points of retrieval accuracy, so these results show mechanisms, not
  statistically robust rates. A real evaluation needs hundreds of questions across
  messy documents.
- **Labels:** one reviewer, no second rater, so there is no inter-rater agreement figure.
  Ablation answers are scored automatically only.
  The automatic check can't see a missing header (A-800 Q1). hit@1 counts any chunk
  holding a fact as relevant, including lures like a headerless table fragment.
- **Answers are noisy even at temperature 0.** A second generation of the identical
  prompts (`results/rerun/`, compared in `results/answer_stability.md`) changed 6–8 of 10
  answer texts per run and moved answer accuracy by up to 10 points. The committed
  run's B-min100 Q4 refusal did not reproduce. Answer-level comparisons need many samples
  per prompt; retrieval, which is deterministic, is the reliable signal here.
- **FAQ boundaries:** §10's bold questions aren't split points, so a question about the
  FAQ retrieves all three answers together.
- **Production next steps:** convert PDF/HTML to Markdown first, so B has headings to work
  with; consider hierarchical (parent-child) retrieval for long sections; use a real
  vector database once there are thousands of chunks; use LLM-as-a-judge or RAGAS once
  there are too many answers to review by hand; add a cross-encoder reranker to push lure
  fragments down.

## 11. Repository layout

```
configs/experiments.toml     every run, models, k, budget, rate limits
data/                        source document (never edited) + frozen answer key
src/rag_chunking/
  chunking/                  fixed.py (A), structure.py + recursive.py (B), base.py
  embeddings/                Gemini embedder, disk cache, rate limiter / retries
  generation/                prompt, cached LLM, Gemini + OpenAI-compatible backends
  evaluation/                answer key, auto-check, review.csv, metrics, report
  retrieval.py               NumPy cosine search, top-k and token-budget modes
  pipeline.py, inspection.py, normalize.py, tokens.py, __main__.py (CLI)
results/<run>/               chunks, retrievals, answers, review.csv, metrics, manifest
results/comparison.md        generated cross-run report
results/review_suggestions.json   suggested labels with reasons (bound to the judged context/answer)
results/rerun/<run>/         a second, independent generation of the same prompts
results/answer_stability.md  generated comparison of the two generations
tests/                       pytest suite (no API calls)
```
