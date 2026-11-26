# Day 12 – Activity 6: Building a RAG Pipeline from Scratch

Module AI04 (Embeddings, Vector Databases & RAG). **Objective:** build the three RAG stages in code (indexing,
retrieval and generation) using SDKs directly, without LangChain or any other RAG framework.

**Learning reference (for the learner to watch; viewing is not claimed here):** *Learn RAG From Scratch – Python AI
Tutorial from a LangChain Engineer*, freeCodeCamp.org, https://www.youtube.com/watch?v=sVcwVQRHIc8, 00:00:00–00:22:14
(chapters: Overview, Indexing, Retrieval, Generation).

## The three functions (`rag.py`)

| Stage | Function | What it does |
|---|---|---|
| **Indexing** | `ingest() -> int` | Loads `documents/*.md`, makes one chunk per `## ` section (ID `<file>::<section>::0`, metadata `source`, `section`, `title`), embeds them with local all-MiniLM-L6-v2 and stores them in the persistent Chroma collection `rag_from_scratch` (`rag_db/`, cosine). Rebuilds on every run, so data is never duplicated. Returns the chunk count. |
| **Retrieval** | `retrieve(query, k=4) -> list[dict]` | Embeds the query and returns the top-k chunks, best first: `id`, `source`, `section`, `text`, `distance`, `similarity = 1 - distance`. Rejects an empty query or `k < 1`. |
| **Generation** | `answer(query, k=4) -> dict` | Retrieves chunks. If none has similarity ≥ 0.30 it returns NOT_FOUND **without calling the LLM**. Otherwise it sends only those chunks to Groq `openai/gpt-oss-120b` (temperature 0), with rules: answer only from the context, cite `[chunk_id]`, reply `NOT_FOUND` if the answer is missing. **Citations are validated:** every cited ID must be one of the chunks sent to the model, otherwise the answer is withheld. |

`answer()` returns `status` (`answered`, `not_found`, `not_found_no_context` or `invalid_citations`), `answer`,
`retrieved` (chunk IDs), `citations` (valid cited IDs) and `invalid` (cited IDs that were not retrieved).

**SDKs used directly:** `chromadb` (vector store, plus its ONNX embedding function) and `groq` (chat completions).
No LangChain. The documents are the 20 synthetic Brightlake policies from Activity 4 (copied, so this folder runs on
its own).

## Setup and run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # chromadb 1.5.9, groq 1.7.0, python-dotenv, pytest
cp .env.example .env                     # GROQ_API_KEY (or rely on ../policy-qa-rag/.env)
python rag.py demo                       # ingest + 5 sample questions -> results.txt
python rag.py retrieve "How many days per week can I work from home?" -k 3
python rag.py ask "What is the hotel limit per night?"
python -m pytest -v                      # 12 offline tests (no API)
```

`rag_db/`, `.env` and `.venv/` are git-ignored.

## Actual results

**Live demo (`results.txt`, 2026-11-26 12:56 UTC, Groq free tier, 5 LLM calls): OVERALL PASS**

The availability check before the run: a tiny 81-token request succeeded (7,919 of 8,000 tokens per minute
remaining). The free tier was used and no paid API.

| # | Question | Top retrieved chunk (similarity) | Status | Cited chunk |
|---|---|---|---|---|
| Q1 | Paid annual leave days for full-time employees? | `leave-01-annual-leave::entitlement::0` (0.814) | answered: "20 days of paid annual leave per calendar year" | `leave-01-annual-leave::entitlement::0` |
| Q2 | Lost the phone with my authenticator app? | `sec-02-multi-factor-authentication::lost-phone::0` (0.767) | answered: "contact the IT Service Desk immediately to reset MFA" | `sec-02-multi-factor-authentication::lost-phone::0` |
| Q3 | Hotel limit per night on business travel? | `work-04-travel-and-expenses::booking-travel::0` (0.573) | answered: "up to USD 180 per night" | `work-04-travel-and-expenses::booking-travel::0` |
| Q4 | Days per week working from home? | `work-01-remote-work::remote-days::0` (0.634) | answered: "up to 2 days per week" | `work-01-remote-work::remote-days::0` |
| Q5 | Policy on bringing pets to the office? *(not in the documents)* | `hr-01-code-of-conduct::respect-at-work::0` (0.311) | **not_found**: the model returned NOT_FOUND | none |

**Citation check:** for all 5 questions, every cited chunk ID was one of the retrieved chunk IDs, with 0 invalid
citations. Q5's best chunk (0.311) was just above the 0.30 threshold, so the LLM was called and correctly refused.
The no-LLM path (`not_found_no_context`) is covered by an offline test.

**Offline tests (`offline-tests.txt`): 12 passed.** They cover ingestion count (60) and no duplicates on re-ingest,
unique IDs and metadata, top-k sorting, bad-input rejection, valid citations accepted, fabricated or missing citations
withheld, NOT_FOUND from the LLM, the irrelevant-query path that skips the LLM, and the prompt containing only
retrieved chunks. The LLM is replaced by test stand-ins, used only in tests.

## Files

| File | Content |
|---|---|
| `rag.py` | `ingest()`, `retrieve(query, k)`, `answer(query)` and the CLI/demo |
| `test_rag.py` | 12 offline tests |
| `documents/` | 20 synthetic Markdown policies |
| `results.txt` | Actual output of the live demo |
| `offline-tests.txt` | Actual pytest output |
| `requirements.txt`, `.env.example`, `.gitignore` | Setup |
| `rag_db/` | Persistent Chroma database (git-ignored) |
