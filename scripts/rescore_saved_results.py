"""Re-check a saved smoke-test result file against the citation-support rule, offline.

Run 1 was scored before the citation-support rule existed. This script makes NO
API calls: it reads the saved responses and citations from the JSON results file,
rebuilds the chunks locally (chunking is deterministic), and checks that every
required fact appears in the text of a chunk the saved answer validly cites.
The saved responses themselves are not modified.

Usage:
    python scripts/rescore_saved_results.py results/smoke-results-run1.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_settings  # noqa: E402
from src.ingestion import chunk_documents, load_documents  # noqa: E402
from src.smoke_eval import load_questions  # noqa: E402

CHUNK_RE = re.compile(r"Chunk:\s*([A-Z]{2,5}-\d{3}-CH\d{2})")


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    path = Path(sys.argv[1])
    saved = json.loads(path.read_text(encoding="utf-8"))
    settings = load_settings()
    chunks = chunk_documents(load_documents(settings.policy_dir), settings.chunk_size, settings.chunk_overlap)
    text_by_id = {c.chunk_id: c.text.lower() for c in chunks}
    questions = {q.id: q for q in load_questions()}

    print(f"Offline citation-support re-check of {path} (no API calls; saved responses unchanged)")
    print(
        f"Chunks rebuilt locally: {len(chunks)} (chunk_size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    print(f"Original verdicts in file: {saved['correct']}/{saved['total']} correct")
    print()
    supported_all = 0
    answerable = 0
    for r in saved["results"]:
        q = questions[r["id"]]
        if not q.answerable:
            print(
                f"{r['id']}: unanswerable - citation support not applicable (citations: {r['valid_citations']})"
            )
            continue
        answerable += 1
        cited = {m for c in r["valid_citations"] for m in CHUNK_RE.findall(c)}
        cited_text = " ".join(text_by_id.get(cid, "") for cid in cited)
        support = [any(alt.lower() in cited_text for alt in group) for group in q.required_facts]
        ok = all(support)
        supported_all += ok
        print(
            f"{r['id']}: cited={sorted(cited)} facts_supported_by_cited_chunks={support} -> {'OK' if ok else 'NOT SUPPORTED'}"
        )
    print()
    print(f"Answerable answers whose citations support every required fact: {supported_all}/{answerable}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
