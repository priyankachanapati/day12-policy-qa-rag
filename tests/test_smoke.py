"""Smoke-test suite checks.

* Structure tests always run (question set composition).
* Integration tests call the real Groq API and are SKIPPED when GROQ_API_KEY is
  not set. The full 15-question run is scripts/run_smoke_test.py.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.config import load_settings
from src.rag_pipeline import RAGPipeline
from src.smoke_eval import evaluate, load_questions

QUESTIONS = load_questions()
HAS_KEY = bool(load_settings().groq_api_key)


def test_exactly_15_questions_with_3_unanswerable():
    assert len(QUESTIONS) == 15
    assert sum(not q.answerable for q in QUESTIONS) == 3
    assert len({q.id for q in QUESTIONS}) == 15


def test_answerable_questions_cover_all_six_policies():
    covered = {q.expected_policy_id for q in QUESTIONS if q.answerable}
    assert covered == {"HR-001", "HR-002", "HR-003", "HR-004", "IT-001", "IT-002"}


def test_answerable_questions_define_required_facts():
    for q in QUESTIONS:
        if q.answerable:
            assert q.required_facts and all(q.required_facts), q.id


def test_required_facts_actually_exist_in_expected_policy(test_settings):
    """Guards against an unfair test: every required fact must be present in the source policy."""
    from src.ingestion import load_documents

    texts = {d.policy_id: d.text.lower() for d in load_documents(test_settings.policy_dir)}
    for q in QUESTIONS:
        for group in q.required_facts:
            assert any(alt.lower() in texts[q.expected_policy_id] for alt in group), (q.id, group)


@pytest.fixture(scope="module")
def live_pipeline(test_settings):
    pipeline = RAGPipeline(dataclasses.replace(test_settings))
    pipeline.ingest()
    return pipeline


@pytest.mark.integration
@pytest.mark.skipif(not HAS_KEY, reason="GROQ_API_KEY not set - live LLM test skipped")
@pytest.mark.parametrize("qid", ["SM-02", "SM-13"])
def test_live_answer_and_refusal(live_pipeline, qid):
    q = next(q for q in QUESTIONS if q.id == qid)
    evaluation = evaluate(q, live_pipeline.ask(q.question))
    assert evaluation.correct, evaluation.reasons
