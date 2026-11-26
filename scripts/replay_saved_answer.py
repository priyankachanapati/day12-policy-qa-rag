"""Replay a SAVED smoke-test answer and re-check it locally - NO LLM/API call.

Shows the answer exactly as the LLM produced it during a saved run, then re-runs
the local parts of the pipeline for the same question: top-k retrieval (local
embeddings + ChromaDB), citation validation against the retrieved chunks, and the
refusal check. Used for evidence screenshots when the LLM quota is unavailable.

Usage:
    python scripts/replay_saved_answer.py results/smoke-results-run1.json SM-02
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.citations import validate_citations  # noqa: E402
from src.config import load_settings  # noqa: E402
from src.embeddings import LocalEmbedder  # noqa: E402
from src.generation import is_refusal  # noqa: E402
from src.retrieval import Retriever, format_retrieval  # noqa: E402
from src.vector_store import PolicyVectorStore  # noqa: E402


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    results_path, test_id = Path(sys.argv[1]), sys.argv[2]
    saved = json.loads(results_path.read_text(encoding="utf-8"))
    record = next((r for r in saved["results"] if r["id"] == test_id), None)
    if record is None:
        print(f"{test_id} not found in {results_path}")
        return 1

    print("=" * 100)
    print(f"REPLAY OF SAVED OUTPUT - NO LLM/API CALL | file: {results_path} | test: {test_id}")
    print(f"Saved run started (UTC): {saved['started_utc']} | LLM in that run: {saved['llm_model']}")
    print("=" * 100)
    print(f"Question: {record['question']}")
    tag = f"[SAVED {results_path.stem} - REPLAY, NO API CALL]"
    print(f"{tag} status : {record['status']}")
    print(f"{tag} answer : {record['raw_llm_answer'] or record['response']}")
    print(f"{tag} verdict: {'CORRECT' if record['correct'] else 'INCORRECT'}")
    print()

    settings = load_settings()
    store = PolicyVectorStore(settings.chroma_dir, settings.collection_name)
    if store.count() == 0:
        print("Vector store is empty - run: python -m src.rag_pipeline ingest")
        return 1
    retrieval = Retriever(LocalEmbedder(), store, settings.top_k, settings.min_similarity).retrieve(
        record["question"]
    )
    print("--- Live local re-check (embeddings + ChromaDB + validator; no LLM) ---")
    print(format_retrieval(retrieval))
    answer = record["raw_llm_answer"] or record["response"]
    report = validate_citations(answer, retrieval.relevant_chunks)
    print(f"Refusal sentence detected : {is_refusal(answer)}")
    print(f"Valid citations           : {', '.join(map(str, report.valid)) or 'none'}")
    print(f"Invalid citations         : {', '.join(str(c) for c, _ in report.invalid) or 'none'}")
    print(f"Malformed citations       : {', '.join(report.malformed) or 'none'}")
    print("=" * 100)
    print(
        f"REPLAY: answer above is SAVED output from {results_path.name} ({saved['llm_model']}, "
        f"{saved['started_utc']}); no LLM/API call was made now."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
