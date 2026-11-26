"""Day 12, Activity 4 - query the persistent Chroma collection (3 queries, metadata filters).

Runs in a separate process from ingest.py, so every result is read from disk. Then
drops the client, reopens the database and repeats query 1 to verify persistence.

Usage:
    python query.py
"""

from __future__ import annotations

import sys

import chromadb
from chromadb.api.client import SharedSystemClient

from ingest import COLLECTION, DB_DIR, EMBEDDING_MODEL, embedding_function

TOP_K = 3
QUERIES = [
    {
        "label": "Q1 - no filter",
        "text": "How many days of annual leave do full-time employees get?",
        "where": None,
    },
    {
        "label": "Q2 - metadata filter on topic",
        "text": "What should I do if I lose my phone?",
        "where": {"topic": "security"},
        "compare_unfiltered": True,
    },
    {
        "label": "Q3 - metadata filter on topic AND section",
        "text": "How much paid time off do I get when I am ill?",
        "where": {"$and": [{"topic": "leave"}, {"section": "Entitlement"}]},
    },
]


def matches(meta: dict, where: dict | None) -> bool:
    """Check a result's metadata against the filter used ($and and equality, as used here)."""
    if not where:
        return True
    if "$and" in where:
        return all(matches(meta, clause) for clause in where["$and"])
    return all(meta.get(k) == v for k, v in where.items())


def open_collection():
    client = chromadb.PersistentClient(path=str(DB_DIR))
    return client, client.get_collection(COLLECTION, embedding_function=embedding_function())


def run_query(collection, text: str, where: dict | None):
    return collection.query(
        query_texts=[text],
        n_results=TOP_K,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def show(res, out) -> list[tuple[str, float]]:
    rows = []
    for rank, (cid, doc, meta, dist) in enumerate(
        zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True), 1
    ):
        body = doc.split("\n", 1)[1]
        out(
            f"  {rank}. distance={dist:.4f} similarity={1 - dist:.4f} | source={meta['source']} | "
            f"section={meta['section']} | topic={meta['topic']}"
        )
        out(f"     {body}")
        rows.append((cid, dist))
    return rows


def main() -> int:
    if not DB_DIR.exists():
        print("chroma_db/ not found - run: python ingest.py", file=sys.stderr)
        return 1
    out = print
    client, collection = open_collection()
    out("=" * 90)
    out("DAY 12 - ACTIVITY 4: QUERIES (separate process - data read from the persistent chroma_db/)")
    out("=" * 90)
    out(
        f"Collection: {collection.name} | chunks on disk: {collection.count()} | embeddings: {EMBEDDING_MODEL}"
    )
    out("Scores: cosine distance (lower = closer); similarity = 1 - distance (higher = closer)")

    all_ok = True
    first_rows = None
    for q in QUERIES:
        out("")
        out(f"{q['label']}: {q['text']!r}")
        out(f"  where = {q['where']}")
        res = run_query(collection, q["text"], q["where"])
        rows = show(res, out)
        if first_rows is None:
            first_rows = rows
        ok = all(matches(m, q["where"]) for m in res["metadatas"][0])
        # A filter can match fewer chunks than TOP_K; then Chroma correctly returns all matches.
        available = (
            len(collection.get(where=q["where"], include=[])["ids"]) if q["where"] else collection.count()
        )
        expected = min(TOP_K, available)
        count_ok = len(rows) == expected
        all_ok &= ok and count_ok
        if q["where"]:
            out(f"  [{'PASS' if ok else 'FAIL'}] all {len(rows)} results satisfy the metadata filter")
            out(
                f"  [{'PASS' if count_ok else 'FAIL'}] {available} chunks in the collection match this filter; "
                f"expected min({TOP_K}, {available}) = {expected} results, got {len(rows)}"
            )
        if q.get("compare_unfiltered"):
            plain = run_query(collection, q["text"], None)
            out("  Same query WITHOUT the filter (for comparison):")
            for rank, meta, dist in zip(
                range(1, TOP_K + 1), plain["metadatas"][0], plain["distances"][0], strict=True
            ):
                out(
                    f"    {rank}. distance={dist:.4f} | source={meta['source']} | section={meta['section']} | topic={meta['topic']}"
                )

    # Persistence: drop the client and its cached system, reopen from disk, repeat Q1.
    del collection, client
    SharedSystemClient.clear_system_cache()
    client2, reopened = open_collection()
    res = run_query(reopened, QUERIES[0]["text"], None)
    again = list(zip(res["ids"][0], res["distances"][0], strict=True))
    same = [cid for cid, _ in again] == [cid for cid, _ in first_rows] and all(
        abs(a[1] - b[1]) < 1e-6 for a, b in zip(again, first_rows, strict=True)
    )
    out("")
    out("Persistence check: client closed, system cache cleared, database reopened from chroma_db/")
    out(f"  reopened collection count: {reopened.count()}")
    out(f"  Q1 after reopen: {[f'{cid} ({d:.4f})' for cid, d in again]}")
    out(f"  [{'PASS' if same else 'FAIL'}] same chunk IDs and distances as before reopening")
    all_ok &= same
    out("")
    out(f"OVERALL: {'PASS' if all_ok else 'FAIL'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
