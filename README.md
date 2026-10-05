# RAG Chunking Strategy Comparison

Comparing fixed-size (naive) chunking against structure-aware chunking with a
recursive fallback, on `data/diabetes_reference_document.md`.

## Layout

```
data/      reference document
src/       chunking, embedding, retrieval, generation, evaluation
results/   evaluation outputs
```

## Setup

```bash
uv sync
cp .env.example .env   # add your GEMINI_API_KEY
```

## Scoring

Expected answers are fixed in `data/answer_key.json` before any retrieval runs, so
judgments can't drift toward a preferred strategy. Each question lists its source
sections and required facts. Each fact has a list of accepted strings.

- **Labels:** yes = 1, partial = 0.5, no = 0.
- **Yes:** at least `min_facts_for_yes` facts are found, and every source section has
  at least one matched fact. This rule only matters for Q5, which needs both §5.1 and §8.
- **Partial:** at least one fact is found, but the yes rule isn't met.
- **Matching:** case-insensitive substring match after unifying dash characters,
  collapsing whitespace, and writing numeric ranges one way ("3 to 6", "3 – 6" → "3-6").
  Accepted strings must appear in the document, so a fact can always be found in a
  chunk. The range rule is there so that a correct answer like "every 3 to 6 months"
  isn't marked wrong for its wording. It is content-blind, applied to chunks and answers
  alike (so it favours neither strategy), and was fixed before any run. Synonyms such as
  "coronary heart disease" are deliberately *not* accepted; those go to manual review.
- **Retrieval accuracy:** the rule is applied to the retrieved chunks, over the 9
  answerable questions. Q9 isn't answered in the document, so it's reported as N/A.
- **Answer accuracy:** the rule is applied to the generated answer, over all 10
  questions. Q9 scores yes only for a correct refusal.
- **Manual review:** the automatic labels are a starting point. Every label is
  reviewed by hand, and overrides are recorded in `review.csv` with a reason.

_Approach, results and analysis: to be added._
