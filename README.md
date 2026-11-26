# Policy Q&A RAG v1

A retrieval-augmented generation (RAG) assistant that answers employee questions about a set of
**synthetic** HR and IT policies, with **validated citations** and **explicit refusal** when the
policies do not contain the answer.

> All policy documents in `data/policies/` are synthetic samples for a fictional company,
> "Brightlake Systems". They are not real policies.

Built for AI Engineering Training, Module AI04 (Embeddings, Vector Databases & RAG), Day 12 lab.

## How it works

```
data/policies/*.md ──► load + parse metadata ──► paragraph-aware chunking (800 chars, 150 overlap)
                                                         │  chunk IDs like HR-001-CH02
                                                         ▼
                       local embeddings: all-MiniLM-L6-v2 (ONNX, 384-d) ──► ChromaDB (cosine)
                                                                                 │
question ──► embed ──► top-k search (k=4) ──► keep chunks with similarity ≥ 0.25 ┘
                                   │
               none relevant? ─────┴──► refuse (LLM not called)
                                   │
                                   ▼
        Groq LLM (openai/gpt-oss-120b) with strict prompt: context-only, cite every fact
        as [Source: HR-001, Chunk: HR-001-CH02], or reply with the fixed refusal sentence
                                   │
                                   ▼
        citation validator: every cited chunk must be one that was retrieved for this
        question and match its policy ID; otherwise the answer is withheld
```

| Component | Choice |
|---|---|
| Language | Python 3.12 |
| Embedding model | `all-MiniLM-L6-v2`, run locally as ONNX through ChromaDB (no API key, 384 dimensions) |
| Vector database | ChromaDB `PersistentClient`, cosine distance, stored in `.chroma/` (git-ignored) |
| LLM | `openai/gpt-oss-120b` through the Groq API, temperature 0 |
| Scores | `distance` = cosine distance (lower is better); `similarity = 1 - distance` (higher is better) |

## Project structure

```
policy-qa-rag/
├── data/policies/            6 synthetic policies (HR-001..HR-004, IT-001, IT-002)
├── src/
│   ├── config.py             settings from environment / .env
│   ├── ingestion.py          loading, front-matter metadata, chunking
│   ├── embeddings.py         local MiniLM embedder
│   ├── vector_store.py       ChromaDB wrapper
│   ├── retrieval.py          top-k retrieval with scores
│   ├── generation.py         prompt, refusal rule, Groq client
│   ├── citations.py          citation extraction + validation
│   ├── rag_pipeline.py       end-to-end pipeline + CLI
│   └── smoke_eval.py         smoke-test correctness rules
├── scripts/
│   ├── run_smoke_test.py         15-question smoke test runner (writes log + JSON)
│   ├── rescore_saved_results.py  offline citation-support re-check of a saved run
│   └── replay_saved_answer.py    show a saved answer + local re-check (no API call)
├── tests/                    pytest suite + smoke_questions.json
├── results/                  saved smoke-test logs and JSON results
├── docs/                     lab documentation and evidence screenshots
├── requirements.txt
├── .env.example
└── pyproject.toml            project metadata + ruff config
```

## Setup

Requires Python 3.10 or newer (developed with 3.12) and a Groq API key (https://console.groq.com/keys).

```bash
cd policy-qa-rag
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # then edit .env and set GROQ_API_KEY
```

The first run downloads the embedding model (about 80 MB) to `~/.cache/chroma`.

## Usage

```bash
# 1. Ingest: load, chunk, embed and store the policies
python -m src.rag_pipeline ingest

# 2. Retrieval only: top-k chunks with similarity and distance
python -m src.rag_pipeline retrieve "How many days of annual leave do I get?" --top-k 4

# 3. Ask: cited answer or refusal
python -m src.rag_pipeline ask "How many unused leave days can I carry forward?"
python -m src.rag_pipeline ask "Does the company offer pet insurance?"
```

## Tests

```bash
# Unit tests and checks (no API key needed; live LLM tests are skipped without a key)
python -m pytest -v

# Lint
ruff check src tests scripts

# Full 15-question smoke test against the real LLM (12 answerable, 3 unanswerable)
python scripts/run_smoke_test.py --log results/smoke-test-log.txt --json results/smoke-results.json
```

Smoke-test correctness rules (`tests/smoke_questions.json`):

* **Answerable:** the answer is not refused, contains every required fact, has at least one citation,
  every citation is valid, at least one citation points to the expected policy, and every required
  fact appears in the text of a chunk the answer cites.
* **Unanswerable:** the fixed refusal sentence, with no citations.

The Groq free tier allows 8,000 tokens per minute, so the runner waits 8 seconds between questions
and the client waits out any `429` response using the `retry-after` header.

## Results

| Run | Log | Result |
|---|---|---|
| Run 1 (official) | `results/smoke-test-log-run1.txt` | **15/15 correct**: 12/12 answered with valid citations, 3/3 refused, 0 invalid citations |
| Run 2 | `results/smoke-test-log-run2-quota-interrupted.txt` | Interrupted by the Groq daily token quota after 9 questions (9/9 correct); not a 15-question result |

Run 3 (final verification) was not performed because of the API quota. See `docs/Day-12-Policy-QA-RAG-Documentation.md`.

## Configuration

All settings can be overridden through environment variables or `.env` (see `.env.example`):
`CHUNK_SIZE`, `CHUNK_OVERLAP`, `TOP_K`, `MIN_SIMILARITY`, `LLM_MODEL`, `LLM_TEMPERATURE`,
`CHROMA_DIR`, `CHROMA_COLLECTION`.

## Known limitations

* Fact checking in the smoke test uses string matching against expected phrasings, so it can miss
  correct answers that are worded differently. It checks that required facts are present, not that
  the answer contains no extra wrong statements.
* The relevance threshold (0.25) is a simple heuristic for MiniLM similarities. Off-topic questions
  that share vocabulary with a policy can pass it, so the LLM's refusal rule is the main defence.
* The LLM output is non-deterministic even at temperature 0, so results can vary between runs.
* The document set is small (6 policies, 22 chunks); behaviour on a large corpus has not been tested.

## Day 12 – Activity 1: Embeddings Deep Dive

`activities/embeddings-deep-dive/` is a separate, self-contained activity. It embeds 20 sentences with the Google
Gemini embeddings API (`gemini-embedding-001`, 3072 dimensions) and prints a 10x10 cosine-similarity matrix, the
closest pair, and the top-3 nearest neighbours for all 20 sentences. It does not use or change the RAG pipeline.

- Closest pair: S01 & S02 ("It is raining heavily outside today." / "There is a lot of rain falling this
  afternoon."), cosine similarity **0.9419**.
- Saved output: `activities/embeddings-deep-dive/results/execution-output.txt`
- Write-up: `docs/Day-12-Activity-1-Embeddings-Deep-Dive.md`; screenshots `docs/evidence/10-*.png`, `11-*.png`
- Run it: see `activities/embeddings-deep-dive/README.md`. The cached embeddings are included, so a run makes no
  API call unless `--refresh` is passed.

## Day 12 – Activity 2: Word2Vec and How Embeddings Are Learned

`activities/word2vec-contextual-embeddings/` checks why "refund" and "reimbursement" sit close together in embedding
space. It reuses Activity 1's embedding and cosine-similarity code by importing it (same Gemini model and settings).

- **cosine(refund, reimbursement) = 0.9154** (`gemini-embedding-001`, 3072 dimensions). "reimbursement" ranks 1st
  of 7 comparison words for "refund"; unrelated controls (banana, thunderstorm) score about 0.77-0.78.
- Saved output: `activities/word2vec-contextual-embeddings/results.txt`; write-up:
  `docs/Day-12-Activity-2-Word2Vec-How-Embeddings-Are-Learned.md`
- Includes the exactly-three-sentence answer, how neural networks learn embeddings from context, and Word2Vec
  versus modern embedding models (see the activity README).
