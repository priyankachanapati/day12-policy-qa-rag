"""Smoke-test question loading and correctness evaluation.

The correctness rules mirror the ``correctness_criteria`` block in
``tests/smoke_questions.json``. They were fixed before run 1; after run 1 one
*stricter* rule was added (citation support: each required fact must appear in
the text of a chunk the answer validly cites) and applied from run 2 onward.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .rag_pipeline import ANSWERED, RAGAnswer

QUESTIONS_FILE = Path(__file__).resolve().parent.parent / "tests" / "smoke_questions.json"


@dataclass(frozen=True)
class SmokeQuestion:
    id: str
    question: str
    answerable: bool
    expected_policy_id: str | None
    expected_answer: str
    required_facts: list[list[str]]


@dataclass
class SmokeEvaluation:
    question: SmokeQuestion
    correct: bool
    retrieval_hit: bool | None  # expected policy among relevant retrieved chunks (answerable only)
    facts_found: list[bool]
    citations_ok: bool
    refusal_ok: bool | None  # refusal behaviour correct (unanswerable only)
    reasons: list[str] = field(default_factory=list)
    # per required-fact group: does a validly cited chunk actually contain the fact?
    citation_support: list[bool] = field(default_factory=list)


def load_questions(path: Path = QUESTIONS_FILE) -> list[SmokeQuestion]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        SmokeQuestion(
            id=q["id"],
            question=q["question"],
            answerable=q["answerable"],
            expected_policy_id=q["expected_policy_id"],
            expected_answer=q["expected_answer"],
            required_facts=q["required_facts"],
        )
        for q in data["questions"]
    ]


def evaluate(q: SmokeQuestion, result: RAGAnswer) -> SmokeEvaluation:
    """Apply the pre-defined correctness criteria to one pipeline result."""
    answer_lc = result.answer.lower()
    report = result.citations
    reasons: list[str] = []

    if q.answerable:
        retrieval_hit = any(c.policy_id == q.expected_policy_id for c in result.retrieval.relevant_chunks)
        facts_found = [any(alt.lower() in answer_lc for alt in group) for group in q.required_facts]
        citations_ok = bool(report and report.all_valid)
        cites_expected = bool(report and q.expected_policy_id in report.cited_policy_ids)
        cited_ids = {c.chunk_id for c in report.valid} if report else set()
        cited_text = " ".join(c.text.lower() for c in result.retrieval.chunks if c.chunk_id in cited_ids)
        citation_support = [any(alt.lower() in cited_text for alt in group) for group in q.required_facts]

        if result.status != ANSWERED:
            reasons.append(f"status is {result.status}, expected answered")
        for group, found in zip(q.required_facts, facts_found, strict=True):
            if not found:
                reasons.append(f"missing required fact (any of {group})")
        if not citations_ok:
            reasons.append("citations missing, invalid or malformed")
        if not cites_expected:
            reasons.append(f"no valid citation to expected policy {q.expected_policy_id}")
        for group, supported in zip(q.required_facts, citation_support, strict=True):
            if not supported:
                reasons.append(f"fact {group} not found in any cited chunk (citation does not support it)")
        if not retrieval_hit:
            reasons.append(f"expected policy {q.expected_policy_id} not retrieved")
        correct = (
            result.status == ANSWERED
            and all(facts_found)
            and citations_ok
            and cites_expected
            and all(citation_support)
        )
        return SmokeEvaluation(
            q, correct, retrieval_hit, facts_found, citations_ok, None, reasons, citation_support
        )

    has_citations = bool(report and report.has_citations)
    refusal_ok = result.is_refusal and not has_citations
    if not result.is_refusal:
        reasons.append(f"status is {result.status}, expected a refusal")
    if has_citations:
        reasons.append("refusal contains citations")
    return SmokeEvaluation(q, refusal_ok, None, [], not has_citations, refusal_ok, reasons)
