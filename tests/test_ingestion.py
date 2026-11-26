"""Tests for document loading, chunking and metadata."""

from __future__ import annotations

import itertools

import pytest

from src.ingestion import chunk_documents, chunk_text, load_documents, parse_front_matter

EXPECTED_IDS = {"HR-001", "HR-002", "HR-003", "HR-004", "IT-001", "IT-002"}


def test_loads_all_six_policies_with_metadata(test_settings):
    docs = load_documents(test_settings.policy_dir)
    assert {d.policy_id for d in docs} == EXPECTED_IDS
    for d in docs:
        assert d.title and d.department and d.source_file.endswith(".md")
        assert "SYNTHETIC" in d.extra_metadata["classification"]
        assert "SYNTHETIC SAMPLE POLICY" in d.text


def test_chunks_have_unique_ids_and_complete_metadata(test_settings):
    docs = load_documents(test_settings.policy_dir)
    chunks = chunk_documents(docs, test_settings.chunk_size, test_settings.chunk_overlap)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    for c in chunks:
        assert c.chunk_id.startswith(c.metadata["policy_id"] + "-CH")
        for key in ("chunk_id", "policy_id", "title", "department", "source_file", "section", "chunk_index"):
            assert c.metadata[key] not in (None, ""), key
        assert c.metadata["char_count"] == len(c.text)


def test_chunks_respect_size_limit(test_settings):
    docs = load_documents(test_settings.policy_dir)
    for c in chunk_documents(docs, test_settings.chunk_size, test_settings.chunk_overlap):
        assert len(c.text) <= test_settings.chunk_size


def test_no_chunk_ends_with_a_dangling_heading(test_settings):
    docs = load_documents(test_settings.policy_dir)
    for c in chunk_documents(docs, test_settings.chunk_size, test_settings.chunk_overlap):
        assert not c.text.splitlines()[-1].lstrip().startswith("#"), c.chunk_id


def test_consecutive_chunks_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 30 for i in range(10))
    chunks = chunk_text(text, chunk_size=300, chunk_overlap=60)
    assert len(chunks) > 1
    for (prev, _), (nxt, _) in itertools.pairwise(chunks):
        # The start of each chunk repeats text from the end of the previous one.
        assert nxt[:20] in prev


def test_every_fact_survives_chunking(test_settings):
    """All source text must appear in at least one chunk (nothing dropped)."""
    docs = load_documents(test_settings.policy_dir)
    chunks = chunk_documents(docs, test_settings.chunk_size, test_settings.chunk_overlap)
    joined = "\n".join(c.text for c in chunks)
    for d in docs:
        for paragraph in d.text.split("\n\n"):
            assert paragraph.strip() in joined


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_config_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("some text", size, overlap)


def test_front_matter_requires_fields():
    with pytest.raises(ValueError):
        parse_front_matter("---\ntitle: X\n---\nbody")
    with pytest.raises(ValueError):
        parse_front_matter("no header at all")
