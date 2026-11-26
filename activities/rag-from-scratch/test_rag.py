"""Offline tests for rag.py - no API calls.

The LLM is replaced by small stand-in functions (test doubles, used only here) so that
retrieval, citation validation and the not-found logic can be tested deterministically.
Live generation is exercised separately by `python rag.py demo`.
"""

from __future__ import annotations

import pytest

import rag


@pytest.fixture(scope="module", autouse=True)
def built_index(tmp_path_factory):
    rag.DB_DIR = tmp_path_factory.mktemp("rag_db")  # isolated database for tests
    return rag.ingest()


def test_ingest_creates_one_chunk_per_section(built_index):
    expected = sum(len(rag.chunk_markdown(f)) for f in rag.DOCS_DIR.glob("*.md"))
    assert built_index == expected == 60
    assert rag.ingest() == 60  # re-ingest does not duplicate


def test_chunks_have_metadata_and_unique_ids():
    chunks = [c for f in sorted(rag.DOCS_DIR.glob("*.md")) for c in rag.chunk_markdown(f)]
    assert len({c[0] for c in chunks}) == len(chunks)
    for cid, _text, meta in chunks:
        assert meta["source"].endswith(".md") and meta["section"] and meta["title"]
        assert rag.CITATION_RE.fullmatch(f"[{cid}]")


def test_retrieve_returns_k_sorted_results():
    hits = rag.retrieve("How many days of annual leave do I get?", k=3)
    assert len(hits) == 3
    assert hits[0]["id"] == "leave-01-annual-leave::entitlement::0"
    sims = [h["similarity"] for h in hits]
    assert sims == sorted(sims, reverse=True)


@pytest.mark.parametrize("q,k", [("", 3), ("   ", 3), ("valid", 0)])
def test_retrieve_rejects_bad_input(q, k):
    with pytest.raises(ValueError):
        rag.retrieve(q, k)


def test_answer_accepts_valid_citation():
    def llm(system, user):  # stand-in: cites the top retrieved chunk
        return "You get 20 days. [leave-01-annual-leave::entitlement::0]"

    r = rag.answer("How many days of annual leave do I get?", llm=llm)
    assert r["status"] == "answered"
    assert r["citations"] == ["leave-01-annual-leave::entitlement::0"]
    assert r["citations"][0] in r["retrieved"]


def test_answer_withholds_fabricated_citation():
    def llm(system, user):
        return "You get 99 days. [leave-03-parental-leave::maternity-leave::0]"  # real ID, but not retrieved here

    r = rag.answer("How many days of annual leave do I get?", k=1, llm=llm)
    assert r["status"] == "invalid_citations"
    assert r["invalid"] == ["leave-03-parental-leave::maternity-leave::0"]


def test_answer_without_citation_is_withheld():
    r = rag.answer("How many days of annual leave do I get?", llm=lambda s, u: "You get 20 days.")
    assert r["status"] == "invalid_citations"


def test_not_found_from_llm():
    r = rag.answer("Can I bring my dog to work?", llm=lambda s, u: rag.NOT_FOUND)
    assert r["status"] == "not_found" and r["answer"] == rag.NOT_FOUND


def test_irrelevant_query_returns_not_found_without_llm():
    def llm(system, user):
        raise AssertionError("LLM must not be called")

    r = rag.answer("What is the capital of Australia?", llm=llm)
    assert r["status"] == "not_found_no_context"


def test_prompt_contains_only_retrieved_chunks():
    seen = {}

    def llm(system, user):
        seen["user"] = user
        return rag.NOT_FOUND

    r = rag.answer("How many days of annual leave do I get?", k=2, llm=llm)
    ids_in_prompt = [
        line.split(": ", 1)[1].rstrip("]")
        for line in seen["user"].splitlines()
        if line.startswith("[chunk_id:")
    ]
    assert set(ids_in_prompt) <= set(r["retrieved"]) and len(ids_in_prompt) <= 2
