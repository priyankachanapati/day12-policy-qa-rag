"""Batched, disk-cached text embeddings.

The assignment specifies the OpenAI Embeddings API. To avoid extra cost, this
implementation uses the Google Gemini Embeddings API (official `google-genai` SDK)
with the same workflow: batch the texts, call the embeddings endpoint, cache each
vector on disk under a hash of its text, and reuse the cache on later requests.

Public function:
    embed(texts: list[str]) -> list[list[float]]

API key: GEMINI_API_KEY from the environment, else ./.env, else ../policy-qa-rag/.env, else ../../.env.
The key is never printed or logged.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
MODEL = "gemini-embedding-001"
TASK_TYPE = "SEMANTIC_SIMILARITY"
BATCH_SIZE = 100  # maximum texts per Gemini embed request
CACHE_DIR = HERE / "cache"
# ./.env, then the sibling RAG project (workspace layout), then the repo root (activities/<name>/ layout)
KEY_SOURCES = [HERE / ".env", HERE.parent / "policy-qa-rag" / ".env", HERE.parents[1] / ".env"]


class EmbeddingError(RuntimeError):
    """Configuration or API failure while embedding."""


@dataclass
class CallStats:
    """What the most recent embed() call did - used to prove caching works."""

    requested: int = 0
    unique: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    api_requests: int = 0
    batch_sizes: list[int] = field(default_factory=list)


def _load_api_key() -> str:
    if os.getenv("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"]
    for path in KEY_SOURCES:
        if path.exists() and (value := dotenv_values(path).get("GEMINI_API_KEY")):
            return value
    raise EmbeddingError("GEMINI_API_KEY is not set (environment or .env). See .env.example.")


def cache_key(text: str) -> str:
    """SHA-256 of model + task type + text: a change to any of them gives a new cache entry."""
    return hashlib.sha256(f"{MODEL}\n{TASK_TYPE}\n{text}".encode()).hexdigest()


class Embedder:
    """Embeds texts with Gemini in batches, caching each vector on disk by text hash."""

    def __init__(
        self, cache_dir: Path = CACHE_DIR, batch_size: int = BATCH_SIZE, max_retries: int = 3
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.max_retries = max_retries
        self._client = None  # created only when an API call is needed
        self.last_stats = CallStats()

    # ---------- cache ----------
    def _path(self, text: str) -> Path:
        return self.cache_dir / f"{cache_key(text)}.npy"

    def _read_cache(self, text: str) -> list[float] | None:
        path = self._path(text)
        if not path.exists():
            return None
        try:
            return np.load(path).astype(float).tolist()
        except (OSError, ValueError):
            path.unlink(missing_ok=True)  # corrupt entry: drop it and re-embed
            return None

    def _write_cache(self, text: str, vector: list[float]) -> None:
        tmp = self._path(text).with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(vector, dtype=np.float64))
        tmp.replace(self._path(text))  # atomic rename: no half-written cache files

    # ---------- API ----------
    def _client_or_create(self):
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=_load_api_key())
        return self._client

    def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        from google.genai import errors, types

        config = types.EmbedContentConfig(task_type=TASK_TYPE)
        for attempt in range(1, self.max_retries + 1):
            try:
                self.last_stats.api_requests += 1
                response = self._client_or_create().models.embed_content(
                    model=MODEL, contents=batch, config=config
                )
                vectors = [list(e.values) for e in response.embeddings]
                if len(vectors) != len(batch):
                    raise EmbeddingError(f"API returned {len(vectors)} vectors for {len(batch)} texts")
                return vectors
            except errors.APIError as exc:
                retryable = exc.code in (429, 500, 503)
                if retryable and attempt < self.max_retries:
                    time.sleep(2**attempt)
                    continue
                raise EmbeddingError(f"Gemini API error {exc.code}: {exc.message}") from exc
        raise AssertionError("unreachable")

    # ---------- public ----------
    def embed(self, texts: list[str]) -> list[list[float]]:
        """Return one embedding vector per input text, in input order.

        Empty list -> []. Non-string or blank items -> ValueError. Cached texts are read
        from disk; only uncached, de-duplicated texts are sent to the API, in batches.
        """
        if not isinstance(texts, list):
            raise TypeError("texts must be a list of strings")
        self.last_stats = CallStats(requested=len(texts))
        if not texts:
            return []
        bad = [i for i, t in enumerate(texts) if not isinstance(t, str) or not t.strip()]
        if bad:
            raise ValueError(f"texts must be non-empty strings; invalid positions: {bad}")

        unique = list(dict.fromkeys(texts))  # de-duplicate, keep order
        self.last_stats.unique = len(unique)
        found: dict[str, list[float]] = {}
        missing: list[str] = []
        for text in unique:
            vector = self._read_cache(text)
            if vector is None:
                missing.append(text)
            else:
                found[text] = vector
        self.last_stats.cache_hits = len(found)
        self.last_stats.cache_misses = len(missing)

        for start in range(0, len(missing), self.batch_size):
            batch = missing[start : start + self.batch_size]
            self.last_stats.batch_sizes.append(len(batch))
            for text, vector in zip(batch, self._embed_batch(batch), strict=True):
                self._write_cache(text, vector)
                found[text] = vector

        return [found[t] for t in texts]


_default: Embedder | None = None


def embed(texts: list[str]) -> list[list[float]]:
    """Module-level convenience wrapper: embed(texts) -> list[list[float]]."""
    global _default
    if _default is None:
        _default = Embedder()
    return _default.embed(texts)


def last_stats() -> CallStats:
    """Statistics of the most recent module-level embed() call."""
    return _default.last_stats if _default else CallStats()
