"""Load the answer key and check it against the source document."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Documents and LLMs disagree on how to write the same fact ("5.7%–6.4%", "5.7% - 6.4%",
# "3 to 6 months"). Folding these into one form stops correct answers failing to match.
# The rule is content-blind and applied to chunks and answers alike, so it favours neither
# strategy. It was fixed before any run (see README, Scoring).
_DASHES = re.compile("[‐-―−]")
_SPACE = re.compile(r"\s+")
_NUMERIC_RANGE = re.compile(r"(\d%?)\s*(?:-|to)\s*(?=\d)")
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)


class AnswerKeyError(ValueError):
    """Raised when the answer key is malformed or disagrees with the document."""


def normalize_for_match(text: str) -> str:
    text = _SPACE.sub(" ", _DASHES.sub("-", text)).strip().lower()
    return _NUMERIC_RANGE.sub(r"\1-", text)


@dataclass(frozen=True)
class Fact:
    name: str
    accept: tuple[str, ...]
    section: str

    def matches(self, text: str) -> bool:
        haystack = normalize_for_match(text)
        return any(normalize_for_match(s) in haystack for s in self.accept)


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    answerable: bool
    source_sections: tuple[str, ...]
    required_facts: tuple[Fact, ...]
    min_facts_for_yes: int
    expected_answer: str
    notes: str


@dataclass(frozen=True)
class AnswerKey:
    version: int
    status: str
    questions: tuple[Question, ...]

    @property
    def answerable(self) -> tuple[Question, ...]:
        return tuple(q for q in self.questions if q.answerable)


def _get(raw: dict[str, Any], key: str, kind: type, where: str) -> Any:
    if key not in raw:
        raise AnswerKeyError(f"[{where}] missing key '{key}'")
    value = raw[key]
    if isinstance(value, bool) and kind is not bool or not isinstance(value, kind):
        raise AnswerKeyError(f"[{where}] '{key}' must be {kind.__name__}")
    return value


def _parse_fact(raw: dict[str, Any], where: str) -> Fact:
    accept = _get(raw, "accept", list, where)
    if not accept or not all(isinstance(s, str) and s.strip() for s in accept):
        raise AnswerKeyError(f"[{where}] 'accept' must be a non-empty list of strings")
    return Fact(
        name=_get(raw, "name", str, where),
        accept=tuple(accept),
        section=_get(raw, "section", str, where),
    )


def _parse_question(raw: dict[str, Any]) -> Question:
    where = str(raw.get("id", "<missing id>"))
    facts = tuple(
        _parse_fact(f, f"{where}.fact{i + 1}")
        for i, f in enumerate(_get(raw, "required_facts", list, where))
    )
    q = Question(
        id=_get(raw, "id", str, where),
        question=_get(raw, "question", str, where),
        answerable=_get(raw, "answerable", bool, where),
        source_sections=tuple(_get(raw, "source_sections", list, where)),
        required_facts=facts,
        min_facts_for_yes=_get(raw, "min_facts_for_yes", int, where),
        expected_answer=_get(raw, "expected_answer", str, where),
        notes=_get(raw, "notes", str, where),
    )
    if not q.answerable:
        if q.required_facts or q.source_sections or q.min_facts_for_yes:
            raise AnswerKeyError(
                f"[{where}] unanswerable questions need no facts, sections or min_facts_for_yes"
            )
        return q
    if not q.required_facts or not q.source_sections:
        raise AnswerKeyError(f"[{where}] answerable questions need facts and source sections")
    if not 1 <= q.min_facts_for_yes <= len(q.required_facts):
        raise AnswerKeyError(f"[{where}] min_facts_for_yes must be in [1, {len(q.required_facts)}]")
    for fact in q.required_facts:
        if fact.section not in q.source_sections:
            raise AnswerKeyError(
                f"[{where}] fact '{fact.name}' cites '{fact.section}', not in source_sections"
            )
    for section in q.source_sections:
        if not any(f.section == section for f in q.required_facts):
            raise AnswerKeyError(f"[{where}] source section '{section}' has no facts")
    return q


def load_answer_key(path: Path) -> AnswerKey:
    if not path.is_file():
        raise AnswerKeyError(f"Answer key not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise AnswerKeyError(f"{path}: invalid JSON: {exc}") from exc

    questions = tuple(_parse_question(q) for q in _get(raw, "questions", list, "top level"))
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        raise AnswerKeyError(f"duplicate question ids in {ids}")
    return AnswerKey(
        version=_get(raw, "version", int, "top level"),
        status=_get(raw, "status", str, "top level"),
        questions=questions,
    )


def section_texts(document: str) -> dict[str, str]:
    """Map heading text (without '#') to the text up to the next heading of any level."""
    matches = list(_HEADING.finditer(document))
    sections = {}
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(document)
        sections[m.group(2)] = document[m.end() : end]
    return sections


def validate_against_document(key: AnswerKey, document: str) -> None:
    """Every cited section must exist, and every accepted string must occur in its section.

    Checking per section (not the whole document) stops a fact from being satisfied by a
    lure elsewhere, e.g. 'neuropathy' in the monitoring section.
    """
    sections = section_texts(document)
    errors = []
    for q in key.questions:
        for section in q.source_sections:
            if section not in sections:
                errors.append(f"{q.id}: no heading '{section}' in the document")
        for fact in q.required_facts:
            body = normalize_for_match(sections.get(fact.section, ""))
            for s in fact.accept:
                if normalize_for_match(s) not in body:
                    errors.append(f"{q.id}: '{s}' ({fact.name}) not found in '{fact.section}'")
    if errors:
        raise AnswerKeyError("answer key does not match the document:\n  " + "\n  ".join(errors))
