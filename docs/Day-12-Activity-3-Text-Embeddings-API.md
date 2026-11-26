# Day 12 – Activity 3: Generating Text Embeddings with OpenAI API

| Item | Detail |
|---|---|
| Module | AI04 - Embeddings, Vector Databases & RAG |
| Assignment-specified provider | OpenAI Embeddings API |
| **Provider actually used** | **Google Gemini Embeddings API** (`google-genai` SDK 3.6.1), used in place of OpenAI to avoid cost |
| Model / dimension | `gemini-embedding-001`, `task_type=SEMANTIC_SIMILARITY`, **3072 dimensions** |
| Code | `Day-12/01-Code/openai-text-embeddings/` (`embeddings.py`, `main.py`) |
| Runs | 2026-11-26 12:31 UTC, 2 executions; 2 API requests in total (both in execution 1) |

## 1. Objective

Generate text embeddings through an embeddings API, and understand how they are used for search, clustering and
recommendations.

## 2. Provider note: OpenAI specified, Gemini used

**Provider note (transparency):** the assignment specifies the **OpenAI Embeddings API**. To avoid extra cost,
this implementation uses the **Google Gemini Embeddings API** (`gemini-embedding-001`, official `google-genai`
SDK) with the existing `GEMINI_API_KEY`. The workflow and concepts are the same: send a batch of texts, get back
one vector per text, cache, and compare with cosine similarity. Only the provider, SDK call, model name and vector
size differ. With OpenAI the call would be `client.embeddings.create(model=..., input=texts)`; here it is
`client.models.embed_content(model=..., contents=texts)`. No OpenAI or Groq API was called.

The folder keeps its assigned name, `openai-text-embeddings/`, so it matches the assignment.

## 3. Video details

- **Video:** *Vector Embeddings Tutorial – Code Your Own AI Assistant with GPT-4 API + LangChain + NLP*,
  freeCodeCamp.org: https://www.youtube.com/watch?v=yfHHvmaMkcA
- **Assigned viewing:** 01:49-14:37 (about 13 minutes).
- **Viewing status:** to be watched and confirmed by the learner. This document does not claim the video was
  watched.

## 4. How an embeddings API works

1. The client sends a list of texts and a model name to the provider's embeddings endpoint.
2. The model returns one fixed-length vector of floats per text. Texts with similar meaning get vectors pointing in
   similar directions.
3. The application stores the vectors, often in a vector database, and compares them with cosine similarity:
   `cosine(A, B) = (A · B) / (|A| × |B|)`.
4. Each call costs time and, with paid providers, money. Batching many texts per call and caching vectors that were
   already computed are therefore standard practice.

## 5. Implementation and function signature

```python
def embed(texts: list[str]) -> list[list[float]]
```

Implemented in `embeddings.py`:

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

## 6. Batching

Only texts that are not already cached are sent, after removing duplicates, in batches of up to 100 per
`embed_content` call. The 20 demo sentences therefore needed **one** request (`batch_sizes=[20]`), not 20.

## 7. Hash-based caching

Each vector is stored as `cache/<sha256>.npy`. The SHA-256 key is computed from the model, task type and text, so
the same text always maps to the same file, while a different model or text gets its own entry. On every call,
cached texts are loaded from disk and only misses go to the API. The `cache/` folder is git-ignored.

## 8. Actual output and cache verification

| | Execution 1 (12:31:40 UTC, empty cache) | Execution 2 (12:31:43 UTC, new process) |
|---|---|---|
| 20 sentences, first pass | 0 cache hits, 20 misses → **1 API request** (batch of 20) | **20 cache hits, 0 API requests** |
| Same 20 sentences again | 20 cache hits, **0 API requests**, vectors identical | 20 cache hits, 0 API requests, vectors identical |
| Search query | 1 miss → 1 API request | 1 cache hit, 0 API requests |
| Embedding dimension | **3072** | 3072 |
| Cache files | 0 → 21 | 21 → 21 |

**Total API requests: 2** (both in execution 1). The second execution produced identical search and recommendation
scores without contacting the API.

Full output of both executions (`results.txt`, verbatim):

```
### EXECUTION 1 - 2026-11-26T12:31:40+00:00 (python main.py)
Provider: Google Gemini Embeddings API via google-genai SDK (used instead of OpenAI to avoid cost)
Model: gemini-embedding-001 | cache dir: cache/ | cache files before this execution: 0
embed([]) -> []  (requested=0 unique=0 cache_hits=0 cache_misses=0 api_requests=0 batch_sizes=[])
Pass 1: embed(20 sentences) -> 20 vectors | requested=20 unique=20 cache_hits=0 cache_misses=20 api_requests=1 batch_sizes=[20]
Embedding vector dimension: 3072
Pass 2: embed(same 20 sentences) -> 20 vectors | requested=20 unique=20 cache_hits=20 cache_misses=0 api_requests=0 batch_sizes=[]
Pass 2 vectors identical to pass 1: True
[PASS] second pass served entirely from cache (0 API requests)

Use case - semantic search | query: 'I want my money back for a broken item' (requested=1 unique=1 cache_hits=0 cache_misses=1 api_requests=1 batch_sizes=[1])
  1. 0.9645  I would like a refund for a damaged product.
  2. 0.8798  Can I return an item and get my money back?
  3. 0.7618  The mobile app crashes every time I open it.
Use case - recommendation | similar to: 'Which laptop is best for video editing?'
  0.8113  Recommend a lightweight notebook for travel.
  0.7315  When is the help desk available to chat?
Error handling | embed(['valid text', '   ']) -> ValueError: texts must be non-empty strings; invalid positions: [1]
Cache files after this execution: 21

### EXECUTION 2 - 2026-11-26T12:31:43+00:00 (python main.py)
Provider: Google Gemini Embeddings API via google-genai SDK (used instead of OpenAI to avoid cost)
Model: gemini-embedding-001 | cache dir: cache/ | cache files before this execution: 21
embed([]) -> []  (requested=0 unique=0 cache_hits=0 cache_misses=0 api_requests=0 batch_sizes=[])
Pass 1: embed(20 sentences) -> 20 vectors | requested=20 unique=20 cache_hits=20 cache_misses=0 api_requests=0 batch_sizes=[]
Embedding vector dimension: 3072
Pass 2: embed(same 20 sentences) -> 20 vectors | requested=20 unique=20 cache_hits=20 cache_misses=0 api_requests=0 batch_sizes=[]
Pass 2 vectors identical to pass 1: True
[PASS] second pass served entirely from cache (0 API requests)
[INFO] pass 1 also made 0 API requests: all 20 vectors came from the disk cache of an earlier execution

Use case - semantic search | query: 'I want my money back for a broken item' (requested=1 unique=1 cache_hits=1 cache_misses=0 api_requests=0 batch_sizes=[])
  1. 0.9645  I would like a refund for a damaged product.
  2. 0.8798  Can I return an item and get my money back?
  3. 0.7618  The mobile app crashes every time I open it.
Use case - recommendation | similar to: 'Which laptop is best for video editing?'
  0.8113  Recommend a lightweight notebook for travel.
  0.7315  When is the help desk available to chat?
Error handling | embed(['valid text', '   ']) -> ValueError: texts must be non-empty strings; invalid positions: [1]
Cache files after this execution: 21
```

## 9. Use cases: search, clustering and recommendations

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

## 10. Code location and evidence

| Item | Location |
|---|---|
| `embed()` implementation | `Day-12/01-Code/openai-text-embeddings/embeddings.py` |
| Demo (20 sentences) | `Day-12/01-Code/openai-text-embeddings/main.py` |
| Actual output, both executions | `Day-12/01-Code/openai-text-embeddings/results.txt` |
| Disk cache, 21 vectors (git-ignored) | `Day-12/01-Code/openai-text-embeddings/cache/` |
| README | `Day-12/01-Code/openai-text-embeddings/README.md` |
