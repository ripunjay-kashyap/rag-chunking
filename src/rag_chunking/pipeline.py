"""One experiment run, stage by stage. Each stage writes its output under results/<run>/."""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

import numpy as np

from rag_chunking.chunking import Chunk, make_chunker
from rag_chunking.config import Config, RunConfig
from rag_chunking.embeddings import GeminiEmbedder
from rag_chunking.evaluation.answer_key import (
    AnswerKey,
    load_answer_key,
    validate_against_document,
)
from rag_chunking.inspection import Cut, cut_summary, find_cuts, summary_stats
from rag_chunking.normalize import normalize
from rag_chunking.results import sha256, write_json
from rag_chunking.retrieval import Index, select_context


@dataclass(frozen=True)
class ChunkedRun:
    document: str
    chunks: list[Chunk]
    stats: dict[str, Any]
    cuts: list[Cut]


def load_document(config: Config) -> str:
    return normalize(config.settings.document.read_text(encoding="utf-8"))


def chunk_run(config: Config, run: RunConfig, document: str) -> ChunkedRun:
    chunks = make_chunker(run.params).chunk(document)
    cuts = find_cuts(chunks, document)
    chunked = ChunkedRun(document, chunks, summary_stats(chunks, document), cuts)
    write_json(
        config.settings.results_dir / run.id / "chunks.json",
        {
            "run_id": run.id,
            "strategy": run.strategy,
            "params": asdict(run.params),
            "document_sha256": sha256(document),
            "stats": chunked.stats,
            "cut_summary": cut_summary(cuts),
            "cuts": [asdict(c) for c in cuts],
            "chunks": [c.to_dict() for c in chunks],
        },
    )
    return chunked


def load_checked_key(config: Config, document: str) -> AnswerKey:
    key = load_answer_key(config.settings.answer_key)
    validate_against_document(key, document)
    return key


def retrieve(
    config: Config, run: RunConfig, key: AnswerKey, index: Index, query_vectors: np.ndarray
) -> dict[str, Any]:
    """Rank every chunk for every question and write retrievals.json."""
    s = config.settings
    questions = []
    for question, vector in zip(key.questions, query_vectors, strict=True):
        ranked = index.search(vector)
        context = select_context(ranked, run.retrieval, s.k, s.token_budget)
        in_context = {h.chunk.id for h in context}
        # Log at least retrieve_n hits (for hit@1 / MRR) and every hit in the context.
        logged = ranked[: max(s.retrieve_n, len(context))]
        questions.append(
            {
                "id": question.id,
                "question": question.question,
                "context_ids": [h.chunk.id for h in context],
                "context_tokens": sum(h.chunk.est_tokens for h in context),
                "hits": [
                    {
                        "rank": h.rank,
                        "chunk_id": h.chunk.id,
                        "score": round(h.score, 6),
                        "est_tokens": h.chunk.est_tokens,
                        "in_context": h.chunk.id in in_context,
                    }
                    for h in logged
                ],
            }
        )
    tokens = [q["context_tokens"] for q in questions]
    result = {
        "run_id": run.id,
        "mode": run.retrieval,
        "k": s.k if run.retrieval == "top_k" else None,
        "token_budget": s.token_budget if run.retrieval == "token_budget" else None,
        "retrieve_n": s.retrieve_n,
        "mean_context_tokens": round(sum(tokens) / len(tokens), 1),
        "questions": questions,
    }
    write_json(s.results_dir / run.id / "retrievals.json", result)
    return result


def _git_commit() -> str:
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def execute(config: Config, run: RunConfig, embedder: GeminiEmbedder) -> dict[str, Any]:
    """Run every implemented stage and write manifest.json.

    The manifest is the run log: it records when the run happened and how many API calls
    it made, so unlike the data files it differs between a cold and a warm-cache run.
    """
    document = load_document(config)
    key = load_checked_key(config, document)
    chunked = chunk_run(config, run, document)

    calls_before = embedder.api_calls
    hits_before, misses_before = embedder.cache.hits, embedder.cache.misses
    chunk_vectors = embedder.embed_documents([c.text for c in chunked.chunks])
    query_vectors = embedder.embed_queries([q.question for q in key.questions])
    for name, vectors in (("chunk", chunk_vectors), ("query", query_vectors)):
        norms = np.linalg.norm(vectors, axis=1)
        if vectors.shape[1] != embedder.config.dimensions or not np.allclose(norms, 1, atol=1e-3):
            raise RuntimeError(f"{name} vectors have shape {vectors.shape} / norms {norms}")

    retrievals = retrieve(config, run, key, Index(chunked.chunks, chunk_vectors), query_vectors)

    manifest = {
        "run_id": run.id,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "run": {**asdict(run), "params": asdict(run.params)},
        "document_sha256": sha256(document),
        "answer_key_sha256": sha256(config.settings.answer_key.read_text(encoding="utf-8")),
        "embedding": {
            k: v
            for k, v in asdict(embedder.config).items()
            if k not in ("api_key_env", "batch_size")
        },
        "llm": {"model": config.llm.model, "temperature": config.llm.temperature},
        "settings": {
            "k": config.settings.k,
            "retrieve_n": config.settings.retrieve_n,
            "token_budget": config.settings.token_budget,
        },
        "chunk_stats": chunked.stats,
        "mean_context_tokens": retrievals["mean_context_tokens"],
        "api_calls": {"embedding": embedder.api_calls - calls_before},
        "embedding_cache": {
            "hits": embedder.cache.hits - hits_before,
            "misses": embedder.cache.misses - misses_before,
        },
    }
    write_json(config.settings.results_dir / run.id / "manifest.json", manifest)
    return manifest
