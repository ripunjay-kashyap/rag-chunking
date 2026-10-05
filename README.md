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

_Approach, results and analysis: to be added._
