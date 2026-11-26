"""Demo for Day 12, Activity 3: embed 20 sentences twice and prove the cache works.

Each execution appends its full output to results.txt with a numbered header, so running
the program twice gives a single file showing the first (API) run and the second (cache)
run. A tiny semantic-search and recommendation demo shows typical uses of embeddings.

Usage:
    python main.py
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

import embeddings
from embeddings import MODEL, EmbeddingError, embed

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results.txt"

SENTENCES = [
    "How do I reset my account password?",
    "I forgot my login credentials and cannot sign in.",
    "Where can I track my parcel delivery?",
    "My order has not arrived yet; where is it?",
    "I would like a refund for a damaged product.",
    "Can I return an item and get my money back?",
    "Do you offer discounts for students?",
    "Is there a promo code for first-time customers?",
    "The mobile app crashes every time I open it.",
    "The application freezes on my phone after the update.",
    "Which laptop is best for video editing?",
    "Recommend a lightweight notebook for travel.",
    "What are your customer support opening hours?",
    "When is the help desk available to chat?",
    "How do I change my delivery address?",
    "Can I update the shipping address on my order?",
    "Is my payment information stored securely?",
    "How do you protect my credit card details?",
    "Do you ship internationally?",
    "Can you deliver to another country?",
]


class Output:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, text: str = "") -> None:
        print(text)
        self.lines.append(text)


def cosine(a: list[float], b: list[float]) -> float:
    va, vb = np.asarray(a), np.asarray(b)
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


def stats_line(s) -> str:
    return (
        f"requested={s.requested} unique={s.unique} cache_hits={s.cache_hits} "
        f"cache_misses={s.cache_misses} api_requests={s.api_requests} batch_sizes={s.batch_sizes}"
    )


def main() -> int:
    run_no = RESULTS.read_text(encoding="utf-8").count("### EXECUTION ") + 1 if RESULTS.exists() else 1
    out = Output()
    out(f"### EXECUTION {run_no} - {datetime.now(UTC).isoformat(timespec='seconds')} (python main.py)")
    out("Provider: Google Gemini Embeddings API via google-genai SDK (used instead of OpenAI to avoid cost)")
    cache_files_before = len(list(embeddings.CACHE_DIR.glob("*.npy")))
    out(f"Model: {MODEL} | cache dir: cache/ | cache files before this execution: {cache_files_before}")

    verdict = False
    # Edge case: empty input makes no API call.
    out(f"embed([]) -> {embed([])}  ({stats_line(embeddings.last_stats())})")

    try:
        # Pass 1 of this execution.
        vectors = embed(SENTENCES)
        s1 = embeddings.last_stats()
        dims = {len(v) for v in vectors}
        out(f"Pass 1: embed(20 sentences) -> {len(vectors)} vectors | {stats_line(s1)}")
        out(f"Embedding vector dimension: {dims.pop() if len(dims) == 1 else sorted(dims)}")

        # Pass 2 of this execution: identical input.
        again = embed(SENTENCES)
        s2 = embeddings.last_stats()
        identical = all(np.array_equal(a, b) for a, b in zip(vectors, again, strict=True))
        out(f"Pass 2: embed(same 20 sentences) -> {len(again)} vectors | {stats_line(s2)}")
        out(f"Pass 2 vectors identical to pass 1: {identical}")
        verdict = s2.api_requests == 0 and s2.cache_hits == 20 and identical
        out(f"[{'PASS' if verdict else 'FAIL'}] second pass served entirely from cache (0 API requests)")
        if s1.api_requests == 0:
            out(
                "[INFO] pass 1 also made 0 API requests: all 20 vectors came from the disk cache of an earlier execution"
            )

        # Use case 1: semantic search.
        query = "I want my money back for a broken item"
        qv = embed([query])[0]
        sq = embeddings.last_stats()
        ranked = sorted(range(len(SENTENCES)), key=lambda i: -cosine(qv, vectors[i]))[:3]
        out("")
        out(f"Use case - semantic search | query: {query!r} ({stats_line(sq)})")
        for r, i in enumerate(ranked, 1):
            out(f"  {r}. {cosine(qv, vectors[i]):.4f}  {SENTENCES[i]}")

        # Use case 2: recommendation ("more like this").
        item = 10  # "Which laptop is best for video editing?"
        recs = sorted(
            (i for i in range(len(SENTENCES)) if i != item), key=lambda i: -cosine(vectors[item], vectors[i])
        )[:2]
        out(f"Use case - recommendation | similar to: {SENTENCES[item]!r}")
        for i in recs:
            out(f"  {cosine(vectors[item], vectors[i]):.4f}  {SENTENCES[i]}")

        # Error handling demo: invalid input is rejected before any API call.
        try:
            embed(["valid text", "   "])
        except ValueError as exc:
            out(f"Error handling | embed(['valid text', '   ']) -> ValueError: {exc}")
    except EmbeddingError as exc:
        out(f"ERROR: {exc}")
        return 1
    finally:
        out(f"Cache files after this execution: {len(list(embeddings.CACHE_DIR.glob('*.npy')))}")
        out("")
        with RESULTS.open("a", encoding="utf-8") as fh:
            fh.write("\n".join(out.lines) + "\n")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
