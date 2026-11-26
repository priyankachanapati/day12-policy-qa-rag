# Day 12 – Activity 3: Generating Text Embeddings (Gemini API in place of OpenAI)

Module AI04 (Embeddings, Vector Databases & RAG). **Objective:** generate text embeddings through an embeddings API
and understand their uses in search, clustering and recommendations.

> **Provider note (transparency):** the assignment specifies the **OpenAI Embeddings API**. To avoid extra cost,
> this implementation uses the **Google Gemini Embeddings API** (`gemini-embedding-001`, official `google-genai`
> SDK) with the existing `GEMINI_API_KEY`. The workflow and concepts are the same: send a batch of texts, get back
> one vector per text, cache, and compare with cosine similarity. Only the provider, SDK call, model name and vector
> size differ. With OpenAI the call would be `client.embeddings.create(model=..., input=texts)`; here it is
> `client.models.embed_content(model=..., contents=texts)`. No OpenAI or Groq API was called.
>
> The folder keeps its assigned name, `openai-text-embeddings/`, so it matches the assignment.

**Learning reference (for the learner to watch; viewing is not claimed here):** *Vector Embeddings Tutorial – Code
Your Own AI Assistant with GPT-4 API + LangChain + NLP*, freeCodeCamp.org,
https://www.youtube.com/watch?v=yfHHvmaMkcA, 01:49–14:37.

## Function

```python
from embeddings import embed

vectors: list[list[float]] = embed(["first text", "second text"])
```

`embed(texts: list[str]) -> list[list[float]]` (in `embeddings.py`):

| Behaviour | How |
|---|---|
| **Batching** | Only the texts *not* already cached are sent, de-duplicated, in batches of up to 100 per `embed_content` call (20 sentences = 1 request) |
| **Hash-based disk cache** | Each vector is saved as `cache/<sha256>.npy`. The hash covers model + task type + text, so changing any of them makes a new entry. Writes are atomic (temp file + rename); a corrupt entry is discarded and re-embedded |
| **Cache reuse** | Cached texts are read from disk with no API call |
| **Order** | Output vectors match input order, including duplicates |
| **Empty input** | `embed([])` returns `[]` with no API call |
| **Invalid input** | A non-list raises `TypeError`; blank or non-string items raise `ValueError` (listing their positions) before any API call |
| **API errors** | `google.genai.errors.APIError` becomes `EmbeddingError` with the HTTP code; 429/500/503 are retried with backoff |
| **Missing key** | Clear `EmbeddingError`; the key is read from `GEMINI_API_KEY` (environment, `./.env` or `../policy-qa-rag/.env`) and never printed |
| **Statistics** | `last_stats()` reports requested, unique, cache_hits, cache_misses, api_requests and batch_sizes for the last call |

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # google-genai 3.6.1, numpy, python-dotenv
python main.py                            # appends its full output to results.txt
python main.py                            # second execution: everything served from cache/
```

`cache/`, `.env` and `.venv/` are git-ignored.

## Actual results (`results.txt`, two executions on 2026-11-26)

| | Execution 1 (12:31:40 UTC, empty cache) | Execution 2 (12:31:43 UTC, new process) |
|---|---|---|
| 20 sentences, first pass | 0 cache hits, 20 misses → **1 API request** (batch of 20) | **20 cache hits, 0 API requests** |
| Same 20 sentences again | 20 cache hits, **0 API requests**, vectors identical | 20 cache hits, 0 API requests, vectors identical |
| Search query | 1 miss → 1 API request | 1 cache hit, 0 API requests |
| Embedding dimension | **3072** | 3072 |
| Cache files | 0 → 21 | 21 → 21 |

**Total API requests: 2** (both in execution 1). The second execution produced identical search and recommendation
scores without contacting the API.

## Use cases demonstrated (actual output)

- **Semantic search.** For the query "I want my money back for a broken item":
  1. "I would like a refund for a damaged product." (0.9645)
  2. "Can I return an item and get my money back?" (0.8798)

  The top result shares no key words with the query ("refund" and "damaged" against "money back" and "broken") yet
  scores highest, which shows the search matches on meaning.
- **Recommendation ("more like this").** For "Which laptop is best for video editing?", the top match is
  "Recommend a lightweight notebook for travel." (0.8113). The second suggestion, the help-desk question (0.7315),
  is weak, because only one other sentence is about laptops.
- **Clustering (concept).** The 20 sentences come in 10 paraphrase pairs. Grouping the vectors, for example with
  k-means or by nearest neighbour, would put each pair together, which is how support tickets are grouped by topic.
  Clustering was not computed in this demo.

## Files

| File | Content |
|---|---|
| `embeddings.py` | `embed()` implementation: batching, hash cache, error handling |
| `main.py` | 20-sentence demonstration; appends output to `results.txt` |
| `results.txt` | Actual output of both executions |
| `requirements.txt`, `.env.example`, `.gitignore` | Setup; `.gitignore` excludes `cache/`, `.env` and `.venv/` |
| `cache/` | 21 cached vectors, `<sha256>.npy` (git-ignored) |
