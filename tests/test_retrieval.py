"""Tests for embedding storage and top-k retrieval (real local embeddings, temporary ChromaDB)."""

from __future__ import annotations

import pytest


def test_store_contains_all_chunks(ingested_pipeline):
    assert ingested_pipeline.store.count() > 0


def test_reingestion_does_not_duplicate(ingested_pipeline):
    before = ingested_pipeline.store.count()
    ingested_pipeline.ingest()
    assert ingested_pipeline.store.count() == before


@pytest.mark.parametrize(
    "question,expected_policy",
    [
        ("How many days of annual leave do I get?", "HR-001"),
        ("What are the core hours when working from home?", "HR-002"),
        ("What is the hotel limit per night on business travel?", "HR-003"),
        ("What is the probation period for new employees?", "HR-004"),
        ("How long must my password be?", "IT-001"),
        ("What should I do if my laptop is stolen?", "IT-002"),
    ],
)
def test_top_result_comes_from_expected_policy(ingested_pipeline, question, expected_policy):
    result = ingested_pipeline.retriever.retrieve(question)
    assert result.chunks[0].policy_id == expected_policy


def test_top_k_scores_and_metadata(ingested_pipeline):
    result = ingested_pipeline.retriever.retrieve("How do I request work from home days?", top_k=3)
    assert len(result.chunks) == 3
    sims = [c.similarity for c in result.chunks]
    assert sims == sorted(sims, reverse=True), "results must be best-first"
    for c in result.chunks:
        assert -1.0 <= c.similarity <= 1.0
        assert c.similarity == pytest.approx(1.0 - c.distance)
        assert c.chunk_id and c.policy_id and c.title and c.source_file and c.section


def test_irrelevant_question_scores_lower_than_relevant(ingested_pipeline):
    relevant = ingested_pipeline.retriever.retrieve("How many sick leave days do employees get?")
    irrelevant = ingested_pipeline.retriever.retrieve("What is the capital of Australia?")
    assert irrelevant.chunks[0].similarity < relevant.chunks[0].similarity


def test_empty_question_rejected(ingested_pipeline):
    with pytest.raises(ValueError):
        ingested_pipeline.retriever.retrieve("   ")
