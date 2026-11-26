"""Tests for the Groq client's rate-limit policy (no network: the API call is replaced)."""

from __future__ import annotations

import groq
import httpx
import pytest

from src import generation
from src.generation import GroqGenerator, QuotaExhaustedError


def _rate_limit_error(retry_after: str) -> groq.RateLimitError:
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(
        429,
        headers={"retry-after": retry_after},
        json={"error": {"message": "Rate limit reached in organization org_ABC123 (TPD)"}},
        request=request,
    )
    return groq.RateLimitError("rate limited", response=response, body=None)


def test_long_retry_after_fails_fast_without_sleeping(monkeypatch):
    gen = GroqGenerator(api_key="test-key-not-real", model="m")
    monkeypatch.setattr(gen, "_complete_once", lambda s, u: (_ for _ in ()).throw(_rate_limit_error("3600")))
    monkeypatch.setattr(generation.time, "sleep", lambda s: pytest.fail("must not sleep"))
    with pytest.raises(QuotaExhaustedError) as info:
        gen.complete("sys", "user")
    assert "org_[redacted]" in str(info.value) and "org_ABC123" not in str(info.value)


def test_short_retry_after_waits_then_succeeds(monkeypatch):
    gen = GroqGenerator(api_key="test-key-not-real", model="m")
    outcomes = [_rate_limit_error("2"), "final answer"]

    def fake_once(system_prompt, user_prompt):
        item = outcomes.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    slept: list[float] = []
    monkeypatch.setattr(gen, "_complete_once", fake_once)
    monkeypatch.setattr(generation.time, "sleep", slept.append)
    assert gen.complete("sys", "user") == "final answer"
    assert slept == [3.0]  # retry-after (2s) + 1s margin
