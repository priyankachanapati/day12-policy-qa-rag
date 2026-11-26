"""Day 12, Activity 2 - Word2Vec and how embeddings are learned.

Checks whether "refund" and "reimbursement" are close in embedding space, reusing the
embedding API call, key loading and cosine-similarity code from Activity 1
(../embeddings-deep-dive/embeddings_deep_dive.py) - same provider, model and settings.

To make the score interpretable, it also embeds comparison words (related, loosely
related, unrelated) and two example sentences that use the words in context.

Usage (from this folder, using Activity 1's virtual environment):
    ../embeddings-deep-dive/.venv/bin/python word_similarity.py            # cached if available
    ../embeddings-deep-dive/.venv/bin/python word_similarity.py --refresh  # new API call

Writes results.txt (identical to what is printed). The API key is never printed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ACTIVITY1 = HERE.parent / "embeddings-deep-dive"
sys.path.insert(0, str(ACTIVITY1))

# Reused from Activity 1 (not copied): API call, key loading, cosine maths, settings.
from embeddings_deep_dive import (  # noqa: E402
    MODEL,
    TASK_TYPE,
    EmbeddingAPIError,
    cosine_matrix,
    embed_texts,
    load_api_key,
)

TARGET = ("refund", "reimbursement")
# (label, text, role) - comparison items that give the target score a scale.
ITEMS: list[tuple[str, str, str]] = [
    ("refund", "refund", "target word"),
    ("reimbursement", "reimbursement", "target word"),
    ("repayment", "repayment", "related: money paid back"),
    ("compensation", "compensation", "related: money for a loss"),
    ("invoice", "invoice", "same domain, different role (a bill)"),
    ("salary", "salary", "same domain, different role (wages)"),
    ("banana", "banana", "unrelated control"),
    ("thunderstorm", "thunderstorm", "unrelated control"),
    ("ctx_refund", "The store gave me a refund after I returned the broken headphones.", "example sentence"),
    ("ctx_reimb", "The company paid my reimbursement for the hotel and taxi costs.", "example sentence"),
    ("ctx_control", "I ate a banana for breakfast this morning.", "unrelated sentence"),
]


def load_or_embed(texts: list[str], refresh: bool) -> tuple[np.ndarray, dict]:
    """Cache embeddings in this folder so re-runs make no API calls."""
    digest = hashlib.sha256(json.dumps([MODEL, TASK_TYPE, texts]).encode()).hexdigest()[:16]
    cache = HERE / f"embeddings-cache-{digest}.npz"
    if cache.exists() and not refresh:
        data = np.load(cache, allow_pickle=False)
        meta = json.loads(str(data["meta"]))
        meta["source"] = (
            f"local cache {cache.name} (from API run at {meta['embedded_at_utc']}; no API call now)"
        )
        return data["vectors"], meta
    key, key_source = load_api_key()
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
    parser = argparse.ArgumentParser(description="refund vs reimbursement similarity check")
    parser.add_argument("--refresh", action="store_true", help="ignore the cache and call the API again")
    args = parser.parse_args()

    labels = [i[0] for i in ITEMS]
    texts = [i[1] for i in ITEMS]
    try:
        vectors, meta = load_or_embed(texts, args.refresh)
    except EmbeddingAPIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    sim = cosine_matrix(vectors)
    idx = {label: n for n, label in enumerate(labels)}
    n, dim = vectors.shape
    lines: list[str] = []
    out = lines.append

    out("=" * 92)
    out("DAY 12 - ACTIVITY 2: WORD2VEC AND HOW EMBEDDINGS ARE LEARNED - refund vs reimbursement")
    out("=" * 92)
    out(f"Embedding model : {meta['model']} (Google Gemini API), task_type={meta['task_type']}")
    out(f"Embeddings from : {meta['source']}")
    out(f"API key source  : {meta.get('key_source', 'n/a')} (value never printed)")
    out(f"Items embedded  : {n} | vector dimension: {dim}")
    out(
        "Code reused from: ../embeddings-deep-dive/embeddings_deep_dive.py (embed_texts, cosine_matrix, load_api_key)"
    )
    out("")
    checks = {
        f"{n} vectors returned": n == len(ITEMS),
        "same dimension for all": vectors.ndim == 2,
        "finite values": bool(np.all(np.isfinite(vectors))),
        "scores within [-1, 1]": bool(np.all((sim >= -1) & (sim <= 1))),
        "self-similarity = 1.0": bool(np.allclose(np.diag(sim), 1.0, atol=1e-6)),
    }
    out("Checks: " + " | ".join(f"[{'PASS' if ok else 'FAIL'}] {k}" for k, ok in checks.items()))
    out("")

    a, b = idx[TARGET[0]], idx[TARGET[1]]
    target = float(sim[a, b])
    out(
        f"RESULT  cosine(refund, reimbursement) = {target:.4f}   (angle ~ {np.degrees(np.arccos(target)):.1f} degrees)"
    )
    out("")

    words = [i for i in ITEMS if i[2] != "example sentence" and i[2] != "unrelated sentence"]
    out("Word comparisons (cosine similarity to each target word):")
    out(f"  {'word':<15} {'role':<40} {'vs refund':>10} {'vs reimbursement':>17}")
    for label, _, role in words:
        if label in TARGET:
            continue
        out(f"  {label:<15} {role:<40} {sim[a, idx[label]]:>10.4f} {sim[b, idx[label]]:>17.4f}")
    others = [lab for lab, _, _ in words if lab != "refund"]
    ranked = sorted(others, key=lambda lab: -sim[a, idx[lab]])
    out(
        f"  Ranking of all words by similarity to 'refund': {', '.join(f'{lab} {sim[a, idx[lab]]:.3f}' for lab in ranked)}"
    )
    out(
        f"  'reimbursement' rank among {len(others)} comparison words for 'refund': {ranked.index('reimbursement') + 1}"
    )
    out("")

    out("Example sentences (words used in context):")
    for lab in ("ctx_refund", "ctx_reimb", "ctx_control"):
        out(f"  {lab:<12}: {texts[idx[lab]]}")
    out(f"  cosine(ctx_refund, ctx_reimb)   = {sim[idx['ctx_refund'], idx['ctx_reimb']]:.4f}")
    out(f"  cosine(ctx_refund, ctx_control) = {sim[idx['ctx_refund'], idx['ctx_control']]:.4f}")
    out(f"  cosine(ctx_reimb, ctx_control)  = {sim[idx['ctx_reimb'], idx['ctx_control']]:.4f}")

    text = "\n".join(lines)
    print(text)
    (HERE / "results.txt").write_text(text + "\n", encoding="utf-8")
    print("\nSaved: results.txt")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
