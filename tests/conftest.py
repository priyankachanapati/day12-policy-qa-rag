"""Shared pytest fixtures."""

from __future__ import annotations

import dataclasses

import pytest

from src.config import load_settings
from src.rag_pipeline import RAGPipeline


class ScriptedLLM:
    """TEST DOUBLE for unit tests only - returns a pre-set reply and records calls.

    It lets us test the pipeline's refusal and citation-validation logic
    deterministically without network access. The real Groq integration is
    exercised separately in test_smoke.py (marked ``integration``) and by
    scripts/run_smoke_test.py.
    """

    model = "scripted-test-double"

    def __init__(self, reply: str = "") -> None:
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        return self.reply


@pytest.fixture(scope="session")
def test_settings(tmp_path_factory):
    """Settings pointing at a throw-away ChromaDB directory."""
    return dataclasses.replace(load_settings(), chroma_dir=tmp_path_factory.mktemp("chroma"))


@pytest.fixture(scope="session")
def scripted_llm():
    return ScriptedLLM()


@pytest.fixture(scope="session")
def ingested_pipeline(test_settings, scripted_llm):
    """A pipeline with all policies ingested into the temporary store, using the scripted LLM."""
    pipeline = RAGPipeline(test_settings, llm=scripted_llm)
    pipeline.ingest()
    return pipeline
