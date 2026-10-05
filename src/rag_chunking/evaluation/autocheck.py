"""Automatic yes / partial / no labels from the frozen answer key.

The same rule labels retrieved chunks and generated answers. The labels are a
reproducible starting point; every one is reviewed by hand in review.csv.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_chunking.evaluation.answer_key import Fact, Question

SCORES = {"yes": 1.0, "partial": 0.5, "no": 0.0}

# The prompt asks for one exact sentence, but models vary it slightly
# ("The documents do not contain this information").
_REFUSAL = re.compile(r"\bdo(?:es)? not contain (?:this|the|that) information\b", re.IGNORECASE)
# A refusal to Q9 that still quotes a glucose value has, in effect, given a target.
_GLUCOSE_VALUE = re.compile(r"\d\s*(?:mg/dl|mmol/l)|\d\s*%", re.IGNORECASE)


@dataclass(frozen=True)
class Check:
    label: str  # "yes" | "partial" | "no" | "N/A"
    matched: tuple[str, ...]  # names of the facts found


def is_refusal(answer: str) -> bool:
    return bool(_REFUSAL.search(answer))


def _label(question: Question, matched: list[Fact]) -> str:
    covered = {f.section for f in matched}
    if len(matched) >= question.min_facts_for_yes and covered >= set(question.source_sections):
        return "yes"
    return "partial" if matched else "no"


def check_retrieval(question: Question, chunk_bodies: list[str]) -> Check:
    """Label the in-context chunks. Facts are matched per chunk, never across a boundary."""
    if not question.answerable:
        return Check("N/A", ())
    matched = [f for f in question.required_facts if any(f.matches(b) for b in chunk_bodies)]
    return Check(_label(question, matched), tuple(f.name for f in matched))


def check_answer(question: Question, answer: str) -> Check:
    refused = is_refusal(answer)
    if not question.answerable:
        ok = refused and not _GLUCOSE_VALUE.search(answer)
        return Check("yes" if ok else "no", ())
    if refused:
        return Check("no", ())
    matched = [f for f in question.required_facts if f.matches(answer)]
    return Check(_label(question, matched), tuple(f.name for f in matched))


def is_relevant(question: Question, chunk_body: str) -> bool:
    """For hit@1 / MRR: a chunk is relevant if it holds at least one required fact."""
    return any(f.matches(chunk_body) for f in question.required_facts)
