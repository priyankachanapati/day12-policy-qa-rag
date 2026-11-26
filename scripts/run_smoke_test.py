"""Run the 15-question smoke test against the real pipeline (local embeddings + Groq LLM).

Usage (from the project root):

    python scripts/run_smoke_test.py --log results/smoke-test-log.txt --json results/smoke-results.json

Everything printed to the terminal is also written to the log file, unchanged.
The script re-ingests the documents first so the run is reproducible.
Exit code: 0 if at least 11/15 tests are correct, 1 if fewer, 2 if the run was
aborted because the LLM provider's quota was exhausted (incomplete, not a result).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.generation import GenerationError, QuotaExhaustedError  # noqa: E402
from src.rag_pipeline import RAGPipeline  # noqa: E402
from src.smoke_eval import evaluate, load_questions  # noqa: E402

PASS_THRESHOLD = 11


class Tee:
    """Write output to the terminal and a log file at the same time."""

    def __init__(self, path: Path | None) -> None:
        self._file = open(path, "w", encoding="utf-8") if path else None  # noqa: SIM115 - closed in close()

    def __call__(self, text: str = "") -> None:
        print(text, flush=True)
        if self._file:
            self._file.write(text + "\n")
            self._file.flush()

    def close(self) -> None:
        if self._file:
            self._file.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, default=None, help="Write the full output to this file")
    parser.add_argument("--json", type=Path, default=None, help="Write machine-readable results here")
    parser.add_argument(
        "--delay",
        type=float,
        default=8.0,
        help="Seconds between questions (keeps a run under Groq's free-tier tokens-per-minute limit)",
    )
    args = parser.parse_args()
    for p in (args.log, args.json):
        if p:
            p.parent.mkdir(parents=True, exist_ok=True)

    out = Tee(args.log)
    settings = load_settings()
    settings.require_groq_key()  # fail fast with a clear message if the key is missing
    questions = load_questions()
    started = datetime.now(timezone.utc)

    out("=" * 78)
    out("POLICY Q&A RAG v1 - 15-QUESTION SMOKE TEST")
    out("=" * 78)
    out(f"Run started (UTC) : {started.isoformat(timespec='seconds')}")
    out(f"Python            : {platform.python_version()} on {platform.system()} {platform.machine()}")
    out(f"LLM               : {settings.llm_model} via Groq API (temperature={settings.llm_temperature})")
    out(
        f"Top-k             : {settings.top_k} | relevance threshold: similarity >= {settings.min_similarity}"
    )
    out(
        f"Questions         : {len(questions)} ({sum(q.answerable for q in questions)} answerable, "
        f"{sum(not q.answerable for q in questions)} unanswerable)"
    )
    out("Correctness criteria (defined before the run):")
    out("  Answerable  : status=answered AND all required facts present AND >=1 citation AND all")
    out("                citations valid (retrieved chunk, matching policy ID) AND cites expected policy")
    out("                AND every required fact appears in a validly cited chunk (citation support;")
    out("                stricter rule added after run 1)")
    out("  Unanswerable: mandated refusal AND no citations")
    out("")

    pipeline = RAGPipeline(settings)
    out(pipeline.ingest().format())
    out("")

    records = []
    aborted_reason = None
    for i, q in enumerate(questions):
        if i:
            time.sleep(args.delay)
        out("-" * 78)
        out(
            f"[{q.id}] {'ANSWERABLE' if q.answerable else 'UNANSWERABLE'} | expected source: "
            f"{q.expected_policy_id or 'none'}"
        )
        out(f"Question : {q.question}")
        out(f"Expected : {q.expected_answer}")
        try:
            result = pipeline.ask(q.question)
        except QuotaExhaustedError as exc:
            # Not an answer-quality failure: the provider refused further requests.
            # Stop instead of recording the remaining questions as wrong answers.
            out(f"ERROR    : {exc}")
            out("RUN ABORTED: provider quota exhausted; remaining questions not run.")
            aborted_reason = str(exc)
            break
        except GenerationError as exc:
            out(f"ERROR    : {exc}")
            out("Verdict  : INCORRECT (pipeline error)")
            records.append(
                {
                    "id": q.id,
                    "question": q.question,
                    "answerable": q.answerable,
                    "error": str(exc),
                    "correct": False,
                }
            )
            continue
        ev = evaluate(q, result)
        retrieved = ", ".join(f"{c.chunk_id}({c.similarity:.3f})" for c in result.retrieval.chunks)
        out(f"Retrieved: {retrieved}")
        out(f"Status   : {result.status}")
        out(f"Response : {result.answer}")
        if result.raw_llm_answer and result.raw_llm_answer != result.answer:
            out(f"Raw LLM  : {result.raw_llm_answer}")
        cits = result.citations
        out(
            f"Citations: valid={[str(c) for c in cits.valid] if cits else []} "
            f"invalid={[str(c) for c, _ in cits.invalid] if cits else []} "
            f"malformed={cits.malformed if cits else []}"
        )
        if q.answerable:
            out(
                f"Checks   : retrieval_hit={ev.retrieval_hit} facts={ev.facts_found} citations_ok={ev.citations_ok}"
                f" citation_supports_facts={ev.citation_support}"
            )
        else:
            out(f"Checks   : refusal_correct={ev.refusal_ok} no_citations={ev.citations_ok}")
        out(
            f"Verdict  : {'CORRECT' if ev.correct else 'INCORRECT'}"
            + (f" - {'; '.join(ev.reasons)}" if ev.reasons and not ev.correct else "")
        )
        records.append(
            {
                "id": q.id,
                "question": q.question,
                "answerable": q.answerable,
                "expected_policy_id": q.expected_policy_id,
                "expected_answer": q.expected_answer,
                "status": result.status,
                "response": result.answer,
                "raw_llm_answer": result.raw_llm_answer,
                "retrieved": [
                    {
                        "chunk_id": c.chunk_id,
                        "policy_id": c.policy_id,
                        "similarity": round(c.similarity, 4),
                        "distance": round(c.distance, 4),
                    }
                    for c in result.retrieval.chunks
                ],
                "valid_citations": [str(c) for c in cits.valid] if cits else [],
                "invalid_citations": [str(c) for c, _ in cits.invalid] if cits else [],
                "malformed_citations": cits.malformed if cits else [],
                "retrieval_hit": ev.retrieval_hit,
                "facts_found": ev.facts_found,
                "citations_ok": ev.citations_ok,
                "citation_support": ev.citation_support,
                "refusal_ok": ev.refusal_ok,
                "correct": ev.correct,
                "reasons": ev.reasons,
            }
        )

    total = len(records)
    correct = sum(r["correct"] for r in records)
    ans = [r for r in records if r["answerable"]]
    unans = [r for r in records if not r["answerable"]]
    out("")
    out("=" * 78)
    out("SUMMARY")
    out("=" * 78)
    out(f"{'ID':<7}{'Type':<14}{'Status':<28}{'Verdict':<10}")
    for r in records:
        out(
            f"{r['id']:<7}{'answerable' if r['answerable'] else 'unanswerable':<14}"
            f"{r.get('status', 'error'):<28}{'CORRECT' if r['correct'] else 'INCORRECT':<10}"
        )
    out("")
    out(f"Correct answers          : {correct}/{total}")
    out(f"Answerable correct       : {sum(r['correct'] for r in ans)}/{len(ans)}")
    out(f"Unanswerable refused     : {sum(r['correct'] for r in unans)}/{len(unans)}")
    out(f"Retrieval hit (answerable): {sum(bool(r.get('retrieval_hit')) for r in ans)}/{len(ans)}")
    out(f"Answers w/ valid citations: {sum(bool(r.get('citations_ok')) for r in ans)}/{len(ans)}")
    out(
        f"Citations support facts  : {sum(all(r.get('citation_support') or [False]) for r in ans)}/{len(ans)}"
    )
    out(f"Invalid citations total  : {sum(len(r.get('invalid_citations', [])) for r in records)}")
    llm = pipeline.llm
    out(
        f"LLM usage                : {getattr(llm, 'calls', '?')} calls, "
        f"{getattr(llm, 'prompt_tokens', '?')} prompt + {getattr(llm, 'completion_tokens', '?')} completion tokens"
    )
    if aborted_reason:
        out(
            f"RESULT                   : INCOMPLETE - aborted after {total}/{len(questions)} questions "
            "(provider quota). Not a valid smoke-test result."
        )
    else:
        verdict = "PASS" if correct >= PASS_THRESHOLD else "FAIL"
        out(f"Pass threshold           : {PASS_THRESHOLD}/{len(questions)} -> {verdict}")
    finished = datetime.now(timezone.utc)
    out(
        f"Run finished (UTC)       : {finished.isoformat(timespec='seconds')} "
        f"({(finished - started).total_seconds():.0f}s)"
    )
    out.close()

    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "started_utc": started.isoformat(timespec="seconds"),
                    "llm_model": settings.llm_model,
                    "top_k": settings.top_k,
                    "min_similarity": settings.min_similarity,
                    "correct": correct,
                    "total": total,
                    "aborted_reason": aborted_reason,
                    "results": records,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    if aborted_reason:
        return 2
    return 0 if correct >= PASS_THRESHOLD else 1


if __name__ == "__main__":
    sys.exit(main())
