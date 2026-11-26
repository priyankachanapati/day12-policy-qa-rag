"""Embeddings Deep Dive - Day 12, Activity 1 (AI04: Embeddings, Vector Databases & RAG).

Embeds 20 short sentences with the Gemini embeddings API and shows how
"similar meaning" becomes "nearby vectors":

* Core activity (sentences S01-S10): 10x10 cosine-similarity matrix and the closest pair.
* Extension (all 20 sentences): 20x20 matrix (saved as CSV) and the top-3 nearest
  neighbours of every sentence, sorted by cosine similarity.

Usage:
    python embeddings_deep_dive.py            # uses cached embeddings if present
    python embeddings_deep_dive.py --refresh  # forces new API calls

API key: GEMINI_API_KEY from the environment, else ./.env, else ../policy-qa-rag/.env, else ../../.env.
The key is never printed; only the file it was loaded from is reported.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from dotenv import dotenv_values

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
MODEL = "gemini-embedding-001"
TASK_TYPE = "SEMANTIC_SIMILARITY"
API_BASE = "https://generativelanguage.googleapis.com/v1beta"
# ./.env, then the sibling RAG project (workspace layout), then the repo root (activities/<name>/ layout)
KEY_SOURCES = [HERE / ".env", HERE.parent / "policy-qa-rag" / ".env", HERE.parents[1] / ".env"]

# (id, theme, sentence). S01-S10 = core activity; S11-S20 = extension.
# Deliberate design: paraphrase pairs that share few words (S01/S02, S03/S04, S06/S14,
# S07/S16, S08/S17) and a word-overlap trap with different meanings (S19/S20: "bank").
SENTENCES: list[tuple[str, str, str]] = [
    ("S01", "weather", "It is raining heavily outside today."),
    ("S02", "weather", "There is a lot of rain falling this afternoon."),
    ("S03", "food", "I cooked pasta with tomato sauce for dinner."),
    ("S04", "food", "We had spaghetti in a red sauce for our evening meal."),
    ("S05", "sports", "The football team won the championship final."),
    ("S06", "technology", "My laptop keeps crashing when I open too many programs."),
    ("S07", "travel", "We booked a flight to Tokyo for our summer holiday."),
    ("S08", "learning", "She is studying hard for her mathematics exam."),
    ("S09", "sports", "The striker scored the winning goal in the last minute."),
    ("S10", "technology", "The new phone has a faster processor and a better camera."),
    ("S11", "weather", "The forecast says it will be sunny and warm tomorrow."),
    ("S12", "food", "Fresh fruit and vegetables are part of a healthy diet."),
    ("S13", "sports", "Tennis players need quick reflexes and good stamina."),
    ("S14", "technology", "My computer froze and I had to restart it."),
    ("S15", "travel", "The train journey through the mountains was beautiful."),
    ("S16", "travel", "Our holiday flight to Japan departs next week."),
    ("S17", "learning", "He is revising for his maths test tomorrow."),
    ("S18", "learning", "Online courses make it easier to learn new skills."),
    ("S19", "nature", "The bank of the river was flooded after the storm."),
    ("S20", "finance", "I need to go to the bank to deposit my salary."),
]
CORE_COUNT = 10
TOP_K_NEIGHBOURS = 3


class EmbeddingAPIError(RuntimeError):
    """Raised when the embeddings API cannot be used."""


def load_api_key() -> tuple[str, str]:
    """Return (key, source description). Never returns or prints the key in messages."""
    if os.getenv("GEMINI_API_KEY"):
        return os.environ["GEMINI_API_KEY"], "environment variable GEMINI_API_KEY"
    for path in KEY_SOURCES:
        if path.exists():
            value = dotenv_values(path).get("GEMINI_API_KEY")
            if value:
                return value, f"GEMINI_API_KEY from {os.path.relpath(path, HERE)}"
    raise EmbeddingAPIError(
        "GEMINI_API_KEY not found. Set it in the environment or in .env (see .env.example)."
    )


def _post(url: str, body: dict, key: str, retries: int = 3) -> dict:
    """POST JSON with the key in a header (not the URL). Retries 429/5xx with backoff."""
    data = json.dumps(body).encode("utf-8")
    for attempt in range(1, retries + 1):
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={"Content-Type": "application/json", "x-goog-api-key": key},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.load(resp)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:300].replace(key, "[redacted]")
            if exc.code in (429, 500, 503) and attempt < retries:
                wait = 2**attempt
                print(
                    f"[retry] HTTP {exc.code}; waiting {wait}s (attempt {attempt + 1}/{retries})",
                    file=sys.stderr,
                )
                time.sleep(wait)
                continue
            raise EmbeddingAPIError(f"HTTP {exc.code} from embeddings API: {detail}") from exc
        except urllib.error.URLError as exc:
            if attempt < retries:
                time.sleep(2**attempt)
                continue
            raise EmbeddingAPIError(f"Network error contacting embeddings API: {exc.reason}") from exc
    raise AssertionError("unreachable")


def embed_texts(texts: list[str], key: str) -> tuple[np.ndarray, str, int]:
    """Embed all texts. Tries one batch request first; falls back to one request per text.

    Returns (matrix of shape [n, dim], method used, number of API requests made).
    """
    model_path = f"models/{MODEL}"
    requests_ = [
        {"model": model_path, "content": {"parts": [{"text": t}]}, "taskType": TASK_TYPE} for t in texts
    ]
    try:
        resp = _post(f"{API_BASE}/{model_path}:batchEmbedContents", {"requests": requests_}, key)
        vectors = [e["values"] for e in resp["embeddings"]]
        method, calls = "batchEmbedContents (1 request)", 1
    except EmbeddingAPIError as exc:
        if "HTTP 404" not in str(exc) and "HTTP 400" not in str(exc):
            raise
        print(
            f"[info] batch endpoint unavailable ({str(exc)[:80]}...); embedding one text per request",
            file=sys.stderr,
        )
        vectors = []
        for r in requests_:
            resp = _post(f"{API_BASE}/{model_path}:embedContent", r, key)
            vectors.append(resp["embedding"]["values"])
        method, calls = f"embedContent ({len(texts)} requests)", len(texts) + 1
    matrix = np.asarray(vectors, dtype=np.float64)
    if matrix.shape[0] != len(texts):
        raise EmbeddingAPIError(f"expected {len(texts)} embeddings, got {matrix.shape[0]}")
    return matrix, method, calls


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity of every pair: dot product of L2-normalised vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("zero-length embedding vector")
    unit = vectors / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def closest_pair(sim: np.ndarray) -> tuple[int, int, float]:
    """Indices and score of the most similar pair of *different* sentences."""
    masked = sim.copy()
    np.fill_diagonal(masked, -np.inf)
    i, j = np.unravel_index(np.argmax(masked), masked.shape)
    return int(min(i, j)), int(max(i, j)), float(sim[i, j])


def least_similar_pair(sim: np.ndarray) -> tuple[int, int, float]:
    i, j = np.unravel_index(np.argmin(sim), sim.shape)
    return int(min(i, j)), int(max(i, j)), float(sim[i, j])


def nearest_neighbours(sim: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
    """For each row, the k most similar other sentences, best first."""
    result = []
    for i in range(sim.shape[0]):
        order = [j for j in np.argsort(-sim[i], kind="stable") if j != i][:k]
        result.append([(int(j), float(sim[i, j])) for j in order])
    return result


def format_matrix(sim: np.ndarray, ids: list[str]) -> str:
    header = "      " + " ".join(f"{i:>6}" for i in ids)
    rows = [f"{ids[r]:<5} " + " ".join(f"{sim[r, c]:6.3f}" for c in range(len(ids))) for r in range(len(ids))]
    return "\n".join([header, *rows])


def load_or_embed(texts: list[str], key_loader, refresh: bool) -> tuple[np.ndarray, dict]:
    """Use cached embeddings when model and sentences are unchanged; otherwise call the API."""
    digest = hashlib.sha256(json.dumps([MODEL, TASK_TYPE, texts]).encode()).hexdigest()[:16]
    cache = RESULTS / f"embeddings-cache-{digest}.npz"
    if cache.exists() and not refresh:
        data = np.load(cache, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        meta["source"] = (
            f"local cache {cache.name} (from API run at {meta['embedded_at_utc']}; no API call now)"
        )
        return data["vectors"], meta
    key, key_source = key_loader()
    vectors, method, calls = embed_texts(texts, key)
    meta = {
        "model": MODEL,
        "task_type": TASK_TYPE,
        "method": method,
        "api_requests": calls,
        "key_source": key_source,
        "embedded_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    np.savez(cache, vectors=vectors, meta=np.array(json.dumps(meta)))
    meta["source"] = f"Gemini API ({method}); cached to {cache.name}"
    return vectors, meta


def main() -> int:
    parser = argparse.ArgumentParser(description="Embeddings Deep Dive")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and call the API again")
    args = parser.parse_args()
    RESULTS.mkdir(exist_ok=True)

    ids = [s[0] for s in SENTENCES]
    themes = [s[1] for s in SENTENCES]
    texts = [s[2] for s in SENTENCES]
    try:
        vectors, meta = load_or_embed(texts, load_api_key, args.refresh)
    except EmbeddingAPIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    sim = cosine_matrix(vectors)
    n, dim = vectors.shape

    print("=" * 96)
    print("DAY 12 - ACTIVITY 1: EMBEDDINGS DEEP DIVE")
    print("=" * 96)
    print(f"Embedding model : {meta['model']} (Google Gemini API), task_type={meta['task_type']}")
    print(f"Embeddings from : {meta['source']}")
    print(f"API key source  : {meta.get('key_source', 'n/a')} (value never printed)")
    print(f"Sentences       : {n} | embedding dimension: {dim}")
    print()

    # ---- Verification checks (computed, not assumed) ----
    norms = np.linalg.norm(vectors, axis=1)
    checks = {
        "20 sentences embedded": n == len(SENTENCES) == 20,
        "all embeddings same dimension": vectors.ndim == 2,
        "no NaN/inf values": bool(np.all(np.isfinite(vectors))),
        "similarity matrix is 20x20": sim.shape == (20, 20),
        "matrix symmetric": bool(np.allclose(sim, sim.T, atol=1e-9)),
        "diagonal = 1.0 (self-similarity)": bool(np.allclose(np.diag(sim), 1.0, atol=1e-6)),
        "all scores within [-1, 1]": bool(np.all((sim >= -1) & (sim <= 1))),
    }
    print("Verification checks:")
    for name, ok in checks.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"  vector L2 norms: min={norms.min():.4f} max={norms.max():.4f}")
    print()

    print("Sentences (S01-S10 = core activity, S11-S20 = extension):")
    for sid, theme, text in SENTENCES:
        print(f"  {sid} [{theme:<10}] {text}")
    print()

    # ---- Core activity: first 10 sentences ----
    core = sim[:CORE_COUNT, :CORE_COUNT]
    print("CORE ACTIVITY - cosine-similarity matrix, sentences S01-S10 (1.000 = same direction)")
    print(format_matrix(core, ids[:CORE_COUNT]))
    ci, cj, cs = closest_pair(core)
    li, lj, ls = least_similar_pair(core)
    print()
    print(f"Closest pair (S01-S10)      : {ids[ci]} & {ids[cj]}  cosine = {cs:.4f}")
    print(f"   {ids[ci]}: {texts[ci]}")
    print(f"   {ids[cj]}: {texts[cj]}")
    print(f"Least similar pair (S01-S10): {ids[li]} & {ids[lj]}  cosine = {ls:.4f}")
    print()

    # ---- Extension: all 20 sentences ----
    neighbours = nearest_neighbours(sim, TOP_K_NEIGHBOURS)
    sorted_ok = all(all(a[1] >= b[1] for a, b in zip(row, row[1:], strict=False)) for row in neighbours)
    print(f"EXTENSION - top-{TOP_K_NEIGHBOURS} nearest neighbours for all 20 sentences (cosine similarity)")
    for i, row in enumerate(neighbours):
        cells = "   ".join(f"{ids[j]} {s:.3f}" for j, s in row)
        same = "yes" if themes[row[0][0]] == themes[i] else "no"
        print(f"  {ids[i]} [{themes[i]:<10}] -> {cells}   (nearest shares theme: {same})")
    print(f"  [{'PASS' if sorted_ok else 'FAIL'}] every neighbour list sorted by descending similarity")
    ai, aj, as_ = closest_pair(sim)
    print()
    print(f"Closest pair overall (S01-S20): {ids[ai]} & {ids[aj]}  cosine = {as_:.4f}")
    print(f"   {ids[ai]}: {texts[ai]}")
    print(f"   {ids[aj]}: {texts[aj]}")

    # Theme statistics: same-theme vs different-theme pairs (all 20).
    same_t, diff_t = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (same_t if themes[i] == themes[j] else diff_t).append(sim[i, j])
    hits = sum(themes[row[0][0]] == themes[i] for i, row in enumerate(neighbours))
    i19, i20 = ids.index("S19"), ids.index("S20")
    print()
    print("Observations computed from the matrix:")
    print(f"  mean cosine, same-theme pairs      : {np.mean(same_t):.4f}  ({len(same_t)} pairs)")
    print(f"  mean cosine, different-theme pairs : {np.mean(diff_t):.4f}  ({len(diff_t)} pairs)")
    print(f"  nearest neighbour shares the theme : {hits}/{n} sentences")
    print(
        f"  'bank' trap S19 (river) vs S20 (money): cosine = {sim[i19, i20]:.4f}; "
        f"S19's nearest = {ids[neighbours[i19][0][0]]}, S20's nearest = {ids[neighbours[i20][0][0]]}"
    )

    # ---- Save result files ----
    for name, block, block_ids in (
        ("similarity-matrix-core-10.csv", core, ids[:CORE_COUNT]),
        ("similarity-matrix-all-20.csv", sim, ids),
    ):
        with open(RESULTS / name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(["id", *block_ids])
            for r, rid in enumerate(block_ids):
                writer.writerow([rid, *(f"{v:.6f}" for v in block[r])])
    (RESULTS / "nearest-neighbours-20.json").write_text(
        json.dumps(
            {
                "model": meta["model"],
                "dimension": dim,
                "neighbours": {
                    ids[i]: [{"id": ids[j], "cosine": round(s, 6)} for j, s in row]
                    for i, row in enumerate(neighbours)
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print()
    print(
        "Saved: results/similarity-matrix-core-10.csv, results/similarity-matrix-all-20.csv, "
        "results/nearest-neighbours-20.json"
    )
    return 0 if all(checks.values()) and sorted_ok else 1


if __name__ == "__main__":
    sys.exit(main())
