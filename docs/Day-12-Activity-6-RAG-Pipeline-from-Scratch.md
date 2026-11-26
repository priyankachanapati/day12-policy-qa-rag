# Day 12 – Activity 6: Building a RAG Pipeline from Scratch

| Item | Detail |
|---|---|
| Module | AI04 - Embeddings, Vector Databases & RAG |
| Code | `Day-12/01-Code/rag-from-scratch/rag.py` (`ingest`, `retrieve`, `answer`) |
| SDKs | `chromadb` 1.5.9 (vector store + local ONNX embeddings), `groq` 1.7.0 (generation). **No LangChain** |
| Models | Embeddings: all-MiniLM-L6-v2 (384-d, local). LLM: `openai/gpt-oss-120b` on Groq (free tier) |
| Result | Live demo: 5/5 checks PASS (4 cited answers, 1 correct NOT_FOUND); offline tests: 12 passed |

## 1. Objective

Build the three RAG stages in code (indexing, retrieval and generation) as `ingest()`, `retrieve(query, k)` and
`answer(query)`, using SDKs directly.

## 2. Video details

- **Video:** *Learn RAG From Scratch – Python AI Tutorial from a LangChain Engineer*, freeCodeCamp.org:
  https://www.youtube.com/watch?v=sVcwVQRHIc8
- **Assigned viewing:** 00:00:00-00:22:14 (chapters: Overview, Indexing, Retrieval, Generation).
- **Viewing status:** to be watched and confirmed by the learner. This document does not claim the video was
  watched.

## 3. Indexing, retrieval and generation (function descriptions)

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

## 4. Code explanation

Indexing (`ingest`):

```python
def ingest() -> int:
    """Load documents/*.md, chunk by section, embed (local MiniLM) and store in Chroma. Returns chunk count."""
    files = sorted(DOCS_DIR.glob("*.md"))
    if not files:
        raise FileNotFoundError(f"no Markdown files in {DOCS_DIR}")
    chunks = [c for f in files for c in chunk_markdown(f)]
    collection = _collection(create=True)  # rebuild: re-running never duplicates
    collection.add(
        ids=[c[0] for c in chunks], documents=[c[1] for c in chunks], metadatas=[c[2] for c in chunks]
    )
    return collection.count()
```

Retrieval (`retrieve`):

```python
def retrieve(query: str, k: int = DEFAULT_K) -> list[dict]:
    """Return the top-k chunks for `query`, best first: id, source, section, text, distance, similarity."""
    if not query or not query.strip():
        raise ValueError("query must not be empty")
    if k < 1:
        raise ValueError("k must be >= 1")
    res = _collection().query(
        query_texts=[query.strip()], n_results=k, include=["documents", "metadatas", "distances"]
    )
    return [
        {
            "id": cid,
            "source": meta["source"],
            "section": meta["section"],
            "text": doc,
            "distance": dist,
            "similarity": 1 - dist,
        }
        for cid, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        )
    ]
```

Generation (`answer`). This builds the context from the retrieved chunks only, calls the LLM, then validates the
citations:

```python
def answer(query: str, k: int = DEFAULT_K, llm=_groq_complete) -> dict:
    """Answer `query` using ONLY retrieved chunks. Returns a dict with:

    status       'answered' | 'not_found' | 'not_found_no_context' | 'invalid_citations'
    answer       text shown to the user
    retrieved    retrieved chunk IDs (best first)
    citations    cited chunk IDs that were actually retrieved
    invalid      cited chunk IDs that were NOT retrieved (must be empty for 'answered')
    """
    chunks = retrieve(query, k)
    retrieved = [c["id"] for c in chunks]
    relevant = [c for c in chunks if c["similarity"] >= MIN_SIMILARITY]
    if not relevant:
        return {
            "status": "not_found_no_context",
            "answer": NOT_FOUND,
            "retrieved": retrieved,
            "citations": [],
            "invalid": [],
        }

    context = "\n\n".join(f"[chunk_id: {c['id']}]\n{c['text']}" for c in relevant)
    safe_q = query.replace("<", "(").replace(">", ")")
    text = llm(SYSTEM_PROMPT, f"<context>\n{context}\n</context>\n\n<question>{safe_q}</question>")
    valid, invalid = validate_citations(text, [c["id"] for c in relevant])
    if text.startswith("NOT_FOUND"):
        status, shown = "not_found", NOT_FOUND
    elif invalid or not valid:
        status, shown = (
            "invalid_citations",
            f"Answer withheld: citations missing or not from retrieved chunks. Raw: {text}",
        )
    else:
        status, shown = "answered", text
    return {"status": status, "answer": shown, "retrieved": retrieved, "citations": valid, "invalid": invalid}
```

## 5. Sample questions, retrieved chunk IDs and cited answers

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

## 6. Execution results and tests

**Offline tests: 12 passed** (`offline-tests.txt`, stand-in LLM, no API). They cover ingestion (60 chunks, no
duplicates), metadata and unique IDs, top-k sorting, bad-input rejection, valid citations accepted, fabricated or
missing citations withheld, NOT_FOUND handling, the irrelevant-query path that skips the LLM, and the prompt
containing only retrieved chunks.

Full live demo output (`results.txt`, verbatim):

```
============================================================================================
DAY 12 - ACTIVITY 6: RAG FROM SCRATCH - demo run 2026-11-26T12:56:30+00:00
============================================================================================
[1] ingest(): 20 documents -> 60 chunks stored in Chroma 'rag_from_scratch' (rag_db/)
    embeddings: all-MiniLM-L6-v2 (local ONNX, 384-d, no API) | chunking: one chunk per '## ' section
[2] retrieve(query, k=4) | [3] answer(query): Groq openai/gpt-oss-120b, temperature 0, relevance threshold similarity >= 0.3

Q1 (answerable): How many days of paid annual leave do full-time employees get?
  retrieved: leave-01-annual-leave::entitlement::0 (0.814), leave-02-sick-leave::entitlement::0 (0.621), leave-04-public-holidays::working-on-a-holiday::0 (0.592), leave-01-annual-leave::how-to-request::0 (0.541)
  status   : answered
  answer   : Full-time employees receive 20 days of paid annual leave per calendar year [leave-01-annual-leave::entitlement::0].
  citations: ['leave-01-annual-leave::entitlement::0'] | invalid (not retrieved): none
  [PASS] every cited chunk ID is one of the retrieved chunk IDs
  [PASS] status matches expectation (answerable)

Q2 (answerable): What should I do if I lose the phone with my authenticator app?
  retrieved: sec-02-multi-factor-authentication::lost-phone::0 (0.767), sec-02-multi-factor-authentication::approved-method::0 (0.422), sec-02-multi-factor-authentication::requirement::0 (0.329), sec-03-phishing-and-incidents::incident-response::0 (0.313)
  status   : answered
  answer   : If you lose the phone with the authenticator app, you should contact the IT Service Desk immediately to reset MFA [sec-02-multi-factor-authentication::lost-phone::0].
  citations: ['sec-02-multi-factor-authentication::lost-phone::0'] | invalid (not retrieved): none
  [PASS] every cited chunk ID is one of the retrieved chunk IDs
  [PASS] status matches expectation (answerable)

Q3 (answerable): What is the hotel limit per night on business travel?
  retrieved: work-04-travel-and-expenses::booking-travel::0 (0.573), work-04-travel-and-expenses::meals::0 (0.470), work-02-office-guidelines::meeting-rooms::0 (0.416), work-02-office-guidelines::desk-booking::0 (0.286)
  status   : answered
  answer   : The hotel reimbursement limit is up to USD 180 per night [work-04-travel-and-expenses::booking-travel::0].
  citations: ['work-04-travel-and-expenses::booking-travel::0'] | invalid (not retrieved): none
  [PASS] every cited chunk ID is one of the retrieved chunk IDs
  [PASS] status matches expectation (answerable)

Q4 (answerable): How many days per week can I work from home?
  retrieved: work-01-remote-work::remote-days::0 (0.634), work-01-remote-work::core-hours::0 (0.420), work-01-remote-work::eligibility::0 (0.409), leave-01-annual-leave::entitlement::0 (0.408)
  status   : answered
  answer   : Eligible employees may work remotely for up to 2 days per week [work-01-remote-work::remote-days::0].
  citations: ['work-01-remote-work::remote-days::0'] | invalid (not retrieved): none
  [PASS] every cited chunk ID is one of the retrieved chunk IDs
  [PASS] status matches expectation (answerable)

Q5 (not in documents): What is the company's policy on bringing pets to the office?
  retrieved: hr-01-code-of-conduct::respect-at-work::0 (0.311), work-02-office-guidelines::clean-desk::0 (0.308), work-02-office-guidelines::desk-booking::0 (0.307), work-02-office-guidelines::meeting-rooms::0 (0.271)
  status   : not_found
  answer   : NOT_FOUND: The documents do not contain the answer to this question.
  citations: none | invalid (not retrieved): none
  [PASS] every cited chunk ID is one of the retrieved chunk IDs
  [PASS] status matches expectation (not in documents)

OVERALL: PASS
```

## 7. Code location and evidence

| Item | Location |
|---|---|
| Implementation | `Day-12/01-Code/rag-from-scratch/rag.py` |
| Offline tests | `Day-12/01-Code/rag-from-scratch/test_rag.py` and `offline-tests.txt` |
| Live output | `Day-12/01-Code/rag-from-scratch/results.txt` |
| Documents | `Day-12/01-Code/rag-from-scratch/documents/` (20 files) |
| README | `Day-12/01-Code/rag-from-scratch/README.md` |
