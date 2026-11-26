"""Day 12, Activity 4 - load, chunk and store 20 Markdown documents in a persistent Chroma collection.

Embeddings: all-MiniLM-L6-v2 (384-d), run locally as ONNX through Chroma's built-in
embedding function - the same model as the Day 12 RAG project. No API calls, no key.

Usage:
    python ingest.py          # rebuilds the collection from documents/ (idempotent)
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

HERE = Path(__file__).resolve().parent
DOCS_DIR = HERE / "documents"
DB_DIR = HERE / "chroma_db"
COLLECTION = "brightlake_policies"
EMBEDDING_MODEL = "all-MiniLM-L6-v2 (ONNX, local via Chroma)"
MAX_CHUNK_CHARS = 400  # a section longer than this is split at paragraph boundaries
OVERLAP_CHARS = 60  # carried from the previous piece when a section is split


def embedding_function() -> ONNXMiniLM_L6_V2:
    return ONNXMiniLM_L6_V2()


def parse_markdown(path: Path) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Return (front-matter metadata, [(section heading, section text), ...])."""
    raw = path.read_text(encoding="utf-8")
    match = re.match(r"^---\n(.*?)\n---\n(.*)$", raw, flags=re.S)
    if not match:
        raise ValueError(f"{path.name}: missing front matter")
    meta = dict(line.split(":", 1) for line in match.group(1).splitlines() if ":" in line)
    meta = {k.strip(): v.strip() for k, v in meta.items()}
    for field in ("title", "topic"):
        if not meta.get(field):
            raise ValueError(f"{path.name}: front matter needs '{field}'")
    sections = []
    for block in re.split(r"^## ", match.group(2), flags=re.M)[1:]:
        heading, _, text = block.partition("\n")
        text = text.strip()
        if text:
            sections.append((heading.strip(), text))
    if not sections:
        raise ValueError(f"{path.name}: no '## ' sections found")
    return meta, sections


def split_section(text: str) -> list[str]:
    """One chunk per section; sections over MAX_CHUNK_CHARS are split by paragraph with overlap."""
    if len(text) <= MAX_CHUNK_CHARS:
        return [text]
    pieces, current = [], ""
    for para in text.split("\n\n"):
        if current and len(current) + len(para) + 2 > MAX_CHUNK_CHARS:
            pieces.append(current)
            current = current[-OVERLAP_CHARS:].split(" ", 1)[-1] + "\n\n" + para
        else:
            current = f"{current}\n\n{para}".strip()
    pieces.append(current)
    return pieces


def build_chunks() -> tuple[list[str], list[str], list[dict], int]:
    ids, docs, metas = [], [], []
    files = sorted(DOCS_DIR.glob("*.md"))
    for path in files:
        meta, sections = parse_markdown(path)
        for heading, text in sections:
            for n, piece in enumerate(split_section(text)):
                slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
                ids.append(f"{path.stem}::{slug}::{n}")
                # Title and section are prepended so short chunks keep their context when embedded.
                docs.append(f"{meta['title']} - {heading}\n{piece}")
                metas.append(
                    {
                        "source": path.name,
                        "section": heading,
                        "topic": meta["topic"],
                        "title": meta["title"],
                        "chunk_index": n,
                    }
                )
    return ids, docs, metas, len(files)


def main() -> int:
    ids, docs, metas, n_files = build_chunks()
    if len(ids) != len(set(ids)):
        print("ERROR: duplicate chunk IDs", file=sys.stderr)
        return 1

    client = chromadb.PersistentClient(path=str(DB_DIR))
    if COLLECTION in [c.name for c in client.list_collections()]:
        client.delete_collection(COLLECTION)  # rebuild so re-running never duplicates data
    collection = client.create_collection(
        name=COLLECTION,
        embedding_function=embedding_function(),
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"description": "Day 12 Activity 4 - synthetic Brightlake policies"},
    )
    collection.add(ids=ids, documents=docs, metadatas=metas)  # Chroma embeds via the collection's function

    print("=" * 90)
    print("DAY 12 - ACTIVITY 4: PERSISTENT VECTOR DATABASE WITH CHROMA - INGESTION")
    print("=" * 90)
    print(f"Chroma version     : {chromadb.__version__} | PersistentClient path: chroma_db/")
    print(f"Collection         : {COLLECTION} (cosine distance)")
    print(f"Embedding function : {EMBEDDING_MODEL} - local, no API calls")
    print(f"Documents loaded   : {n_files} Markdown files from documents/")
    print(
        f"Chunking           : one chunk per '## ' section; split at paragraphs if > {MAX_CHUNK_CHARS} chars"
    )
    split = sum(m["chunk_index"] > 0 for m in metas)
    print(f"Chunks stored      : {collection.count()} (sections split into several chunks: {split})")
    print(f"Per topic          : {dict(sorted(Counter(m['topic'] for m in metas).items()))}")
    print(f"Metadata keys      : {sorted(metas[0])}")
    sample = collection.get(ids=[ids[0]], include=["documents", "metadatas", "embeddings"])
    print(f"Embedding dimension: {len(sample['embeddings'][0])}")
    print(f"Sample chunk id    : {ids[0]}")
    print(f"Sample metadata    : {sample['metadatas'][0]}")
    print(f"Sample text        : {sample['documents'][0][:110]}...")
    return 0 if collection.count() == len(ids) else 1


if __name__ == "__main__":
    sys.exit(main())
