"""Tests for citation extraction and validation against the retrieved chunks."""

from __future__ import annotations

from src.citations import extract_citations, validate_citations
from src.rag_pipeline import ANSWERED, REJECTED
from src.retrieval import RetrievedChunk


def _chunk(chunk_id: str, policy_id: str) -> RetrievedChunk:
    return RetrievedChunk(1, chunk_id, policy_id, "T", "S", "f.md", "text", 0.3, 0.7)


CONTEXT = [_chunk("HR-001-CH02", "HR-001"), _chunk("HR-001-CH03", "HR-001")]


def test_extracts_unique_citations_in_order():
    text = "A [Source: HR-001, Chunk: HR-001-CH02]. B [Source: HR-001, Chunk: HR-001-CH02]."
    assert [c.chunk_id for c in extract_citations(text)] == ["HR-001-CH02"]


def test_valid_citation_passes():
    report = validate_citations("20 days. [Source: HR-001, Chunk: HR-001-CH02]", CONTEXT)
    assert report.all_valid and not report.invalid


def test_fabricated_chunk_id_is_invalid():
    report = validate_citations("20 days. [Source: HR-001, Chunk: HR-001-CH09]", CONTEXT)
    assert not report.all_valid
    assert "not in the retrieved context" in report.invalid[0][1]


def test_policy_chunk_mismatch_is_invalid():
    report = validate_citations("x [Source: IT-001, Chunk: HR-001-CH02]", CONTEXT)
    assert not report.all_valid and report.invalid


def test_malformed_citation_detected():
    report = validate_citations("x [Source: HR-001 chunk 2] [Source: HR-001, Chunk: HR-001-CH02]", CONTEXT)
    assert report.malformed == ["[Source: HR-001 chunk 2]"]
    assert not report.all_valid


def test_answer_without_citations_is_not_valid():
    report = validate_citations("Employees get 20 days of leave.", CONTEXT)
    assert not report.has_citations and not report.all_valid
    assert report.uncited_sentences


def test_pipeline_withholds_answer_with_fabricated_citation(ingested_pipeline, scripted_llm):
    scripted_llm.reply = "You get 99 days. [Source: HR-001, Chunk: HR-001-CH99]"
    result = ingested_pipeline.ask("How many days of annual leave do I get?")
    assert result.status == REJECTED
    assert "99 days" not in result.answer


def test_pipeline_accepts_answer_citing_retrieved_chunk(ingested_pipeline, scripted_llm):
    retrieved = ingested_pipeline.retriever.retrieve("How many days of annual leave do I get?")
    top = retrieved.relevant_chunks[0]
    scripted_llm.reply = f"You get 20 days. [Source: {top.policy_id}, Chunk: {top.chunk_id}]"
    result = ingested_pipeline.ask("How many days of annual leave do I get?")
    assert result.status == ANSWERED
    assert result.citations.cited_policy_ids == {top.policy_id}


def test_smoke_eval_rejects_citation_that_does_not_support_fact(ingested_pipeline, scripted_llm):
    """A citation to a real, retrieved chunk that does not contain the fact must not count."""
    from src.smoke_eval import evaluate, load_questions

    q = next(q for q in load_questions() if q.id == "SM-01")  # 20 days of annual leave
    retrieved = ingested_pipeline.retriever.retrieve(q.question).relevant_chunks
    wrong = next(c for c in retrieved if "20 days" not in c.text.lower())
    scripted_llm.reply = f"Employees receive 20 days. [Source: {wrong.policy_id}, Chunk: {wrong.chunk_id}]"
    evaluation = evaluate(q, ingested_pipeline.ask(q.question))
    assert evaluation.citations_ok  # the citation is structurally valid...
    assert not evaluation.correct  # ...but does not support the fact
    assert evaluation.citation_support == [False]
