"""Offline tests for the relevance threshold - no API calls.

Retrieval uses the real local embeddings and Chroma database (Activity 6's rag_db/).
Generation is replaced by stand-in functions (used only in these tests); live generation
is exercised by `python retriever.py`.
"""

from __future__ import annotations

import pytest

import retriever as r


@pytest.fixture(scope="module", autouse=True)
def vector_store():
    """Build Activity 6's local Chroma index if it is missing (fresh clone); no API calls."""
    if not r.rag.DB_DIR.exists():
        r.rag.ingest()


@pytest.fixture(scope="module")
def threshold():
    t, _ = r.calibrate_threshold()
    return t


def no_llm(system, user):
    raise AssertionError("LLM must not be called for a below-threshold question")


def test_similarity_is_one_minus_distance():
    hits = r.rag.retrieve("How long must my password be?", 4)
    for h in hits:
        assert h["similarity"] == pytest.approx(1 - h["distance"])
    # best first: highest similarity == lowest distance
    assert hits[0]["similarity"] == max(h["similarity"] for h in hits)
    assert hits[0]["distance"] == min(h["distance"] for h in hits)


def test_calibration_separates_sets(threshold):
    _, rows = r.calibrate_threshold()
    in_scope = [s for _, k, s in rows if k == "in-scope"]
    off = [s for _, k, s in rows if k == "off-topic"]
    assert max(off) < threshold <= min(in_scope)


def test_test_questions_are_not_calibration_questions():
    calib = set(r.CALIBRATION_IN_SCOPE) | set(r.CALIBRATION_OFF_TOPIC)
    assert not calib & {q for q, _ in r.TEST_QUESTIONS}


@pytest.mark.parametrize("q", [q for q, kind in r.TEST_QUESTIONS if kind == "off-topic"])
def test_off_topic_says_i_dont_know_without_llm(q, threshold):
    res = r.grounded_answer(q, threshold, llm=no_llm)
    assert res["status"] == "i_dont_know" and res["answer"] == r.I_DONT_KNOW
    assert res["best_similarity"] < threshold and not res["llm_called"]


@pytest.mark.parametrize("q", [q for q, kind in r.TEST_QUESTIONS if kind == "in-scope"])
def test_in_scope_passes_threshold_and_only_strong_chunks_reach_llm(q, threshold):
    seen = {}

    def llm(system, user):
        seen["user"] = user
        top = res_ids[0]
        return f"Grounded answer. [{top}]"

    res_ids = [h["id"] for h in r.rag.retrieve(q, r.K) if h["similarity"] >= threshold]
    res = r.grounded_answer(q, threshold, llm=llm)
    assert res["passed"] and res["status"] == "answered"
    assert all(h["similarity"] >= threshold for h in res["context"])
    assert res["citations"] == [res_ids[0]]
    assert all(f"[chunk_id: {cid}]" in seen["user"] for cid in res_ids)


def test_citation_to_unretrieved_chunk_is_withheld(threshold):
    res = r.grounded_answer(
        "When is the IT Service Desk available?",
        threshold,
        llm=lambda s, u: "Open 24/7. [leave-03-parental-leave::maternity-leave::0]",
    )
    assert res["status"] == "invalid_citations"
