# Day 12 – Activity 4: Persistent Vector Database with Chroma

Module AI04 (Embeddings, Vector Databases & RAG). **Objective:** load and chunk documents and store their embeddings
in a persistent Chroma database; then run similarity queries with metadata filters.

**Learning reference (for the learner to watch; viewing is not claimed here):** *RAG + Langchain Python Project:
Easy AI/Chat For Your Docs*, pixegami, https://www.youtube.com/watch?v=tcqEUSNCn8I, 00:00–09:38. This
implementation uses the **ChromaDB Python API directly, without LangChain**.

## What Chroma does here

Chroma is a vector database. It stores, for each chunk, the text, its embedding vector and a metadata dictionary.
At query time it embeds the question, finds the nearest vectors (cosine distance, HNSW index) and can restrict
results with a `where` metadata filter. `PersistentClient(path=...)` writes everything to disk, so the collection
survives restarts.

## Setup and run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt     # chromadb==1.5.9 (same version as the Day 12 RAG project)
python ingest.py                    # load, chunk, embed, store -> chroma_db/
python query.py                     # 3 queries + persistence check (separate process)
```

`chroma_db/` is git-ignored and rebuilt by `ingest.py`. Re-running ingestion deletes and recreates the collection,
so data is never duplicated.

## Design

| Item | Choice |
|---|---|
| Documents | **20** synthetic Markdown files in `documents/`, 4 per topic: `hr`, `it`, `leave`, `security`, `workplace` |
| Document format | front matter (`title`, `topic`) + `# Title` + a synthetic-sample notice + three `## Section` blocks |
| Chunking | one chunk per `## ` section; sections over 400 characters are split at paragraphs with 60-character overlap (none of these short sections needed splitting) |
| Chunk text | `"<title> - <section>\n<section text>"` (the title and section give short chunks context) |
| Metadata | **`source`** (file name), **`section`** (heading), `topic`, `title`, `chunk_index` |
| Chunk IDs | `<file stem>::<section slug>::<n>` (unique, stable) |
| Embeddings | `all-MiniLM-L6-v2`, 384-d, run locally as ONNX through Chroma's embedding function. This is the same model as the RAG project, with **no API calls and no key** |
| Collection | `brightlake_policies`, `configuration={"hnsw": {"space": "cosine"}}` (current Chroma 1.x API) |
| Scores | cosine **distance** (lower = closer); similarity = 1 − distance |

## Actual results (`results.txt`)

Ingestion: **20 documents → 60 chunks** (12 per topic), embedding dimension 384.

| Query | Filter (`where`) | Top result (distance) |
|---|---|---|
| Q1 "How many days of annual leave do full-time employees get?" | none | `leave-01-annual-leave.md` / Entitlement (0.1977): "...20 days of paid annual leave..." |
| Q2 "What should I do if I lose my phone?" | `{"topic": "security"}` | `sec-02-multi-factor-authentication.md` / Lost Phone (0.6050) |
| Q3 "How much paid time off do I get when I am ill?" | `{"$and": [{"topic": "leave"}, {"section": "Entitlement"}]}` | `leave-02-sick-leave.md` / Entitlement (0.4871): "10 days of paid sick leave" |

**Metadata filter checks:**
- Every Q2 and Q3 result satisfies its filter (PASS).
- **Q2 without the filter** returns IT chunks in 2nd and 3rd place (laptop "Lost or Stolen Devices" at 0.6834, Service
  Desk contact at 0.8050). With the filter, only security chunks are returned.
- **Q3** returns 2 results, not 3, because only 2 chunks in the whole collection match `topic=leave AND
  section=Entitlement`. The script checks this against Chroma's own count (PASS).

**Persistence:**
- `query.py` runs in a new process, so all of its data is read from `chroma_db/`.
- It then drops the client, clears Chroma's system cache, reopens the database and repeats Q1: count 60, with the same
  chunk IDs and distances (PASS). **OVERALL: PASS.**

**Transparency note:** the first run of `query.py` printed `OVERALL: FAIL` because of a bug in my check, not in
Chroma. The check required exactly 3 results even when a filter matches fewer chunks. The check was fixed and both
scripts re-run. All retrieved chunks and distances were identical in both runs. The first output is kept as
`results-first-run-check-bug.txt`.

## Files

| File | Content |
|---|---|
| `documents/` | 20 Markdown policy documents |
| `ingest.py` | Load, chunk, embed and store in the persistent collection |
| `query.py` | 3 queries (2 with metadata filters), filter verification, persistence check |
| `results.txt` | Actual output of `ingest.py` and `query.py` (final run) |
| `results-first-run-check-bug.txt` | First run, with the check bug described above |
| `requirements.txt`, `.gitignore` | Setup; `chroma_db/` and `.venv/` are git-ignored |
| `chroma_db/` | Persistent Chroma database (git-ignored) |
