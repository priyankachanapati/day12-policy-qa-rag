"""Day 12, Activity 7 - grounded retrieval with a relevance-score threshold.

Reuses Activity 6 (../rag-from-scratch/rag.py) by importing it: the same Chroma database
(rag_db/, 60 chunks), local all-MiniLM-L6-v2 embeddings, retrieve(), grounded system
prompt, Groq call and citation validator.

Score metric (important):
  * Chroma returns COSINE DISTANCE   d in [0, 2]   - LOWER  = more relevant.
  * We convert to COSINE SIMILARITY  s = 1 - d      - HIGHER = more relevant.
  * The threshold T is defined on SIMILARITY: a question passes if the BEST chunk has
    s >= T (equivalently d <= 1 - T). Otherwise the bot answers "I don't know" and the
    LLM is NOT called.

T is calibrated on a separate set of questions (not the test questions): the midpoint
between the lowest in-scope best-similarity and the highest off-topic best-similarity.

Usage:
    python retriever.py      # calibrate, then run the 4 test questions; saves results.txt
"""

from __future__ import annotations

import sys
import time
from datetime import UTC, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rag-from-scratch"))

import rag  # noqa: E402  (Activity 6: retrieve, prompt, Groq call, citation validator)

I_DONT_KNOW = "I don't know. The policy documents do not contain information relevant to this question."
K = 4

# Calibration questions - used ONLY to choose the threshold.
CALIBRATION_IN_SCOPE = [
    "How many days of paid annual leave do full-time employees get?",
    "What should I do if I lose the phone with my authenticator app?",
    "What is the hotel limit per night on business travel?",
    "How many days per week can I work from home?",
    "How long must my password be?",
    "How often are fire drills held?",
]
CALIBRATION_OFF_TOPIC = [
    "What is the capital of Australia?",
    "How do I learn to play the guitar?",
    "What will the weather be like tomorrow?",
    "What is the current share price of Apple?",
    "Which planet is closest to the sun?",
    "What is the company's policy on bringing pets to the office?",  # near-topic distractor
]

# Test questions - different from the calibration set.
TEST_QUESTIONS = [
    ("When is the IT Service Desk available?", "in-scope"),
    ("How do I report a phishing email?", "in-scope"),
    ("What is a good recipe for chocolate cake?", "off-topic"),
    ("Who won the football World Cup in 2018?", "off-topic"),
]


def best_similarity(query: str, k: int = K) -> tuple[float, float, list[dict]]:
    """Return (best similarity, best distance, hits) for a query. Hits are sorted best first."""
    hits = rag.retrieve(query, k)
    return hits[0]["similarity"], hits[0]["distance"], hits


def calibrate_threshold() -> tuple[float, list[tuple[str, str, float]]]:
    """Midpoint between the lowest in-scope and highest off-topic best similarity."""
    rows = [(q, "in-scope", best_similarity(q)[0]) for q in CALIBRATION_IN_SCOPE]
    rows += [(q, "off-topic", best_similarity(q)[0]) for q in CALIBRATION_OFF_TOPIC]
    low_in = min(s for _, kind, s in rows if kind == "in-scope")
    high_off = max(s for _, kind, s in rows if kind == "off-topic")
    if low_in <= high_off:
        raise RuntimeError(
            f"calibration sets overlap (lowest in-scope {low_in:.4f} <= highest off-topic {high_off:.4f})"
        )
    return round((low_in + high_off) / 2, 2), rows


def retrieve_with_threshold(query: str, threshold: float, k: int = K) -> dict:
    """Retrieve top-k chunks and apply the threshold to the BEST chunk's similarity."""
    best_sim, best_dist, hits = best_similarity(query, k)
    passed = best_sim >= threshold
    return {
        "query": query,
        "hits": hits,
        "best_similarity": best_sim,
        "best_distance": best_dist,
        "threshold": threshold,
        "passed": passed,
        "context": [h for h in hits if h["similarity"] >= threshold],  # only chunks above T go to the LLM
    }


def grounded_answer(query: str, threshold: float, k: int = K, llm=None) -> dict:
    """Answer only from chunks above the threshold, with [chunk_id] citations; else "I don't know"."""
    r = retrieve_with_threshold(query, threshold, k)
    if not r["passed"]:
        return {
            **r,
            "status": "i_dont_know",
            "answer": I_DONT_KNOW,
            "citations": [],
            "invalid": [],
            "llm_called": False,
        }
    llm = llm or rag._groq_complete
    context = "\n\n".join(f"[chunk_id: {h['id']}]\n{h['text']}" for h in r["context"])
    safe_q = query.replace("<", "(").replace(">", ")")
    text = llm(rag.SYSTEM_PROMPT, f"<context>\n{context}\n</context>\n\n<question>{safe_q}</question>")
    valid, invalid = rag.validate_citations(text, [h["id"] for h in r["context"]])
    if text.startswith("NOT_FOUND"):
        status, shown = "i_dont_know", I_DONT_KNOW
    elif invalid or not valid:
        status, shown = "invalid_citations", f"Answer withheld (citation check failed). Raw: {text}"
    else:
        status, shown = "answered", text
    return {
        **r,
        "status": status,
        "answer": shown,
        "citations": valid,
        "invalid": invalid,
        "llm_called": True,
    }


def main() -> int:
    lines: list[str] = []

    def out(s: str = "") -> None:
        print(s, flush=True)
        lines.append(s)

    if not rag.DB_DIR.exists():
        rag.ingest()  # local only (no API)
    out("=" * 96)
    out(
        f"DAY 12 - ACTIVITY 7: GROUNDED RETRIEVAL WITH A RELEVANCE THRESHOLD - {datetime.now(UTC).isoformat(timespec='seconds')}"
    )
    out("=" * 96)
    out(
        "Reused from Activity 6: rag.retrieve (Chroma rag_db/, 60 chunks, all-MiniLM-L6-v2 local), grounded prompt,"
    )
    out("Groq call (openai/gpt-oss-120b) and citation validator.")
    out(
        "Score metric: Chroma cosine DISTANCE d (lower = closer) -> cosine SIMILARITY s = 1 - d (higher = closer)."
    )
    out(
        'Rule: answer only if the BEST chunk has s >= T (i.e. d <= 1 - T); otherwise reply "I don\'t know" (no LLM call).'
    )
    out("")

    threshold, rows = calibrate_threshold()
    out("STEP 1 - CALIBRATION (separate questions, used only to choose T)")
    for q, kind, s in sorted(rows, key=lambda r: -r[2]):
        out(f"  {s:.4f}  {kind:<9}  {q}")
    low_in = min(s for _, k_, s in rows if k_ == "in-scope")
    high_off = max(s for _, k_, s in rows if k_ == "off-topic")
    out(
        f"  lowest in-scope = {low_in:.4f} | highest off-topic = {high_off:.4f} | "
        f"T = midpoint rounded = {threshold:.2f}  (distance cut-off = {1 - threshold:.2f})"
    )
    out("")

    out("STEP 2 - TEST QUESTIONS (not used for calibration)")
    ok_all = True
    llm_calls = 0
    for i, (q, kind) in enumerate(TEST_QUESTIONS, 1):
        if llm_calls:
            time.sleep(8)  # stay under the Groq free-tier tokens-per-minute limit
        out("")
        out(f"T{i} ({kind}): {q}")
        try:
            r = grounded_answer(q, threshold)
        except RuntimeError as exc:
            out(f"  ERROR: {exc}")
            out("  -> generation NOT executed for this question (live API unavailable).")
            ok_all = False
            continue
        llm_calls += r["llm_called"]
        out(
            "  retrieved: "
            + ", ".join(f"{h['id']} (s={h['similarity']:.4f}, d={h['distance']:.4f})" for h in r["hits"])
        )
        out(
            f"  best similarity = {r['best_similarity']:.4f} (distance {r['best_distance']:.4f}) vs T = {threshold:.2f} -> "
            f"{'PASS: retrieve + generate' if r['passed'] else 'BELOW THRESHOLD: I dont know (LLM not called)'}"
        )
        if r["passed"]:
            out(f"  chunks sent to LLM (s >= T): {[h['id'] for h in r['context']]}")
        out(f"  status: {r['status']} | LLM called: {r['llm_called']}")
        out(f"  final answer: {r['answer']}")
        if kind == "in-scope":
            cites_ok = (
                r["status"] == "answered"
                and bool(r["citations"])
                and not r["invalid"]
                and all(c in [h["id"] for h in r["context"]] for c in r["citations"])
            )
            out(f"  citations: {r['citations']} | invalid: {r['invalid'] or 'none'}")
            out(
                f"  [{'PASS' if cites_ok else 'FAIL'}] grounded answer with citations that are all retrieved chunks"
            )
            ok_all &= cites_ok
        else:
            dk = r["status"] == "i_dont_know" and not r["llm_called"]
            out(
                f'  [{"PASS" if dk else "FAIL"}] off-topic question answered "I don\'t know" without calling the LLM'
            )
            ok_all &= dk
    out("")
    out(f"LLM calls made: {llm_calls} (only questions that passed the threshold)")
    out(f"OVERALL: {'PASS' if ok_all else 'FAIL / INCOMPLETE'}")
    (HERE / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Saved: results.txt")
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
