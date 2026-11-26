# Embeddings Deep Dive (Day 12, Activity 1)

AI Engineering Training, Module AI04 (Embeddings, Vector Databases & RAG).

This script embeds 20 short sentences with the **Google Gemini embeddings API** (`gemini-embedding-001`, 3072
dimensions, `task_type=SEMANTIC_SIMILARITY`) and shows that sentences with similar meaning have similar vectors:

- **Core activity (S01-S10):** 10x10 cosine-similarity matrix and the closest pair.
- **Extension (S01-S20):** 20x20 matrix and the top-3 nearest neighbours of every sentence, with scores.

## Setup

```bash
cd embeddings-deep-dive
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env      # add GEMINI_API_KEY (or rely on ../policy-qa-rag/.env)
```

The script looks for `GEMINI_API_KEY` in the environment, then in `./.env`, then in `../policy-qa-rag/.env`. It
prints only which source was used, never the key itself.

## Run

```bash
python embeddings_deep_dive.py            # uses cached embeddings if present (no API call)
python embeddings_deep_dive.py --refresh  # calls the API again (1 batch request for all 20 sentences)
```

All 20 sentences are embedded in **one** `batchEmbedContents` request. The vectors are cached in
`results/embeddings-cache-<hash>.npz`, so later runs make no API calls unless the sentences or model change, or you
pass `--refresh`.

## Outputs (`results/`)

| File | Content |
|---|---|
| `execution-output.txt` | Full terminal output of the API run (2026-11-26 11:33 UTC) |
| `similarity-matrix-core-10.csv` | 10x10 cosine matrix, S01-S10 |
| `similarity-matrix-all-20.csv` | 20x20 cosine matrix |
| `nearest-neighbours-20.json` | Top-3 neighbours with cosine scores for each sentence |
| `embeddings-cache-*.npz` | Raw 20x3072 vectors and run metadata (no secrets) |

## Built-in verification

The script checks, and prints PASS or FAIL for each:
- 20 embeddings of equal dimension, with no NaN or infinite values
- a 20x20 matrix that is symmetric, has 1.0 on the diagonal, and has every score in [-1, 1]
- every neighbour list sorted by descending similarity

It exits with a non-zero code if any check fails.
