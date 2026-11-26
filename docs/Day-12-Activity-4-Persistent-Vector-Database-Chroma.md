# Day 12 – Activity 4: Persistent Vector Database with Chroma

| Item | Detail |
|---|---|
| Module | AI04 - Embeddings, Vector Databases & RAG |
| Code | `Day-12/01-Code/chroma-vector-database/` (`ingest.py`, `query.py`, `documents/`) |
| Database | ChromaDB 1.5.9 `PersistentClient` at `chroma_db/`, collection `brightlake_policies`, cosine distance |
| Embeddings | `all-MiniLM-L6-v2` (384-d), local ONNX via Chroma; **no API calls** |
| Result | 20 documents, 60 chunks; 3 queries (2 filtered); filter and persistence checks PASS |

## 1. Objective

Load and chunk documents, store their embeddings in a persistent Chroma database, and run similarity queries that
use metadata filters.

## 2. Video details

- **Video:** *RAG + Langchain Python Project: Easy AI/Chat For Your Docs*, pixegami:
  https://www.youtube.com/watch?v=tcqEUSNCn8I
- **Assigned viewing:** 00:00-09:38 (about 10 minutes).
- **Viewing status:** to be watched and confirmed by the learner. This document does not claim the video was
  watched.
- The implementation uses the ChromaDB API directly. LangChain is not needed for this activity.

## 3. What Chroma is and how it is used

Chroma is a vector database. It stores, for each chunk, the text, its embedding vector and a metadata dictionary.
At query time it embeds the question, finds the nearest vectors (cosine distance, HNSW index) and can restrict
results with a `where` metadata filter. `PersistentClient(path=...)` writes everything to disk, so the collection
survives restarts.

## 4. Document structure

20 synthetic Markdown files, 4 per topic (HR, IT, leave, security, workplace), for the fictional company Brightlake
Systems. Each file has front matter, a title, a synthetic-sample notice and three `##` sections. For example,
`documents/leave-02-sick-leave.md`:

```
---
title: Sick Leave
topic: leave
---

# Sick Leave

> SYNTHETIC SAMPLE DOCUMENT for training. Brightlake Systems is a fictional company.

## Entitlement

Employees receive 10 days of paid sick leave per calendar year. Unused sick leave does not carry forward.

## Notification

Employees must tell their line manager before 10:00 on the first day of absence.

## Medical Certificate

A medical certificate is required for any sick absence longer than 2 consecutive working days.
```

## 5. Chunking, metadata and ingestion

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

Ingestion output (`python ingest.py`) is shown in Section 8.

## 6. Collection persistence

- `ingest.py` writes the collection to `chroma_db/` and exits.
- `query.py` runs as a **new process** and opens the same path, so every result in Section 7 was read from disk.
- `query.py` then deletes its client, clears Chroma's system cache, reopens the database and re-runs Q1. The count
  (60), chunk IDs and distances were identical (PASS).

## 7. Three queries and filtered results

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

## 8. Actual output (results.txt, verbatim)

```
$ python ingest.py
==========================================================================================
DAY 12 - ACTIVITY 4: PERSISTENT VECTOR DATABASE WITH CHROMA - INGESTION
==========================================================================================
Chroma version     : 1.5.9 | PersistentClient path: chroma_db/
Collection         : brightlake_policies (cosine distance)
Embedding function : all-MiniLM-L6-v2 (ONNX, local via Chroma) - local, no API calls
Documents loaded   : 20 Markdown files from documents/
Chunking           : one chunk per '## ' section; split at paragraphs if > 400 chars
Chunks stored      : 60 (sections split into several chunks: 0)
Per topic          : {'hr': 12, 'it': 12, 'leave': 12, 'security': 12, 'workplace': 12}
Metadata keys      : ['chunk_index', 'section', 'source', 'title', 'topic']
Embedding dimension: 384
Sample chunk id    : hr-01-code-of-conduct::purpose::0
Sample metadata    : {'title': 'Code of Conduct', 'source': 'hr-01-code-of-conduct.md', 'chunk_index': 0, 'section': 'Purpose', 'topic': 'hr'}
Sample text        : Code of Conduct - Purpose
The code of conduct sets the standard of behaviour expected from every employee. It ...
[ingest.py exit=0]

$ python query.py   (new process)
==========================================================================================
DAY 12 - ACTIVITY 4: QUERIES (separate process - data read from the persistent chroma_db/)
==========================================================================================
Collection: brightlake_policies | chunks on disk: 60 | embeddings: all-MiniLM-L6-v2 (ONNX, local via Chroma)
Scores: cosine distance (lower = closer); similarity = 1 - distance (higher = closer)

Q1 - no filter: 'How many days of annual leave do full-time employees get?'
  where = None
  1. distance=0.1977 similarity=0.8023 | source=leave-01-annual-leave.md | section=Entitlement | topic=leave
     Full-time employees receive 20 days of paid annual leave per calendar year. Part-time employees receive leave on a pro-rata basis.
  2. distance=0.3854 similarity=0.6146 | source=leave-04-public-holidays.md | section=Working on a Holiday | topic=leave
     Employees who work on a public holiday receive one compensatory day off, to be taken within 60 days.
  3. distance=0.3902 similarity=0.6098 | source=leave-02-sick-leave.md | section=Entitlement | topic=leave
     Employees receive 10 days of paid sick leave per calendar year. Unused sick leave does not carry forward.

Q2 - metadata filter on topic: 'What should I do if I lose my phone?'
  where = {'topic': 'security'}
  1. distance=0.6050 similarity=0.3950 | source=sec-02-multi-factor-authentication.md | section=Lost Phone | topic=security
     If the phone with the authenticator app is lost, contact the IT Service Desk immediately to reset MFA.
  2. distance=0.8223 similarity=0.1777 | source=sec-03-phishing-and-incidents.md | section=Incident Response | topic=security
     The security team isolates affected devices, resets credentials and informs the employee of the next steps.
  3. distance=0.8291 similarity=0.1709 | source=sec-03-phishing-and-incidents.md | section=Reporting | topic=security
     Suspicious emails must be reported with the Report Phishing button. Suspected account compromise must be reported to the IT Service Desk within 1 hour.
  [PASS] all 3 results satisfy the metadata filter
  [PASS] 12 chunks in the collection match this filter; expected min(3, 12) = 3 results, got 3
  Same query WITHOUT the filter (for comparison):
    1. distance=0.6050 | source=sec-02-multi-factor-authentication.md | section=Lost Phone | topic=security
    2. distance=0.6834 | source=it-01-laptop-and-equipment.md | section=Lost or Stolen Devices | topic=it
    3. distance=0.8050 | source=it-03-service-desk.md | section=Contact | topic=it

Q3 - metadata filter on topic AND section: 'How much paid time off do I get when I am ill?'
  where = {'$and': [{'topic': 'leave'}, {'section': 'Entitlement'}]}
  1. distance=0.4871 similarity=0.5129 | source=leave-02-sick-leave.md | section=Entitlement | topic=leave
     Employees receive 10 days of paid sick leave per calendar year. Unused sick leave does not carry forward.
  2. distance=0.5722 similarity=0.4278 | source=leave-01-annual-leave.md | section=Entitlement | topic=leave
     Full-time employees receive 20 days of paid annual leave per calendar year. Part-time employees receive leave on a pro-rata basis.
  [PASS] all 2 results satisfy the metadata filter
  [PASS] 2 chunks in the collection match this filter; expected min(3, 2) = 2 results, got 2

Persistence check: client closed, system cache cleared, database reopened from chroma_db/
  reopened collection count: 60
  Q1 after reopen: ['leave-01-annual-leave::entitlement::0 (0.1977)', 'leave-04-public-holidays::working-on-a-holiday::0 (0.3854)', 'leave-02-sick-leave::entitlement::0 (0.3902)']
  [PASS] same chunk IDs and distances as before reopening

OVERALL: PASS
[query.py exit=0]
```

## 9. Code location and evidence

| Item | Location |
|---|---|
| Documents | `Day-12/01-Code/chroma-vector-database/documents/` (20 `.md` files) |
| Ingestion | `Day-12/01-Code/chroma-vector-database/ingest.py` |
| Queries and checks | `Day-12/01-Code/chroma-vector-database/query.py` |
| Actual output | `Day-12/01-Code/chroma-vector-database/results.txt` |
| First run (check bug) | `Day-12/01-Code/chroma-vector-database/results-first-run-check-bug.txt` |
| Persistent database (git-ignored) | `Day-12/01-Code/chroma-vector-database/chroma_db/` |
| README | `Day-12/01-Code/chroma-vector-database/README.md` |
