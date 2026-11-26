# Day 12 – Activity 2: Word2Vec and How Embeddings Are Learned

Module AI04 (Embeddings, Vector Databases & RAG). **Objective:** explain how a neural network learns embeddings from
context, and check that "refund" and "reimbursement" are close in embedding space.

**Learning reference (for the learner to watch; viewing is not claimed here):** *Word Embedding and Word2Vec,
Clearly Explained!!!*, StatQuest with Josh Starmer, https://www.youtube.com/watch?v=viZrOnJclY0, 00:00–11:44.

## The assignment answer, in exactly three sentences

"Refund" and "reimbursement" appear in very similar contexts, surrounded by words such as money, paid, back,
receipt, claim and expense, even though a refund usually follows a returned purchase and a reimbursement usually
covers a cost someone paid on another's behalf. An embedding model is trained to predict words from their
surroundings, so words that keep the same company are nudged toward the same region of vector space. Our similarity
script confirms this: the cosine similarity between the two words is 0.9154, the highest of the seven comparison
words tested against "refund".

## How a neural network learns embeddings from context

1. **Start random.** Every word in the vocabulary gets a vector of random numbers.
2. **Make a prediction task from plain text.** Word2Vec slides a small window over sentences. In the *skip-gram*
   version, the network sees one word (for example "refund") and must predict its neighbours ("issued", "full",
   "receipt"). In *CBOW* it does the reverse, predicting the middle word from its neighbours.
3. **Learn from mistakes.** Each wrong prediction produces an error. Backpropagation then adjusts the word's
   vector, and the output weights, slightly in the direction that would have made the right neighbours more likely.
4. **Repeat millions of times.** Words that keep predicting the same neighbours receive similar updates again and
   again, so their vectors drift toward each other. The vector table *is* the embedding. Nobody labels meanings; the
   meaning comes from the context ("you shall know a word by the company it keeps").

## Implementation

`word_similarity.py` **reuses Activity 1's code by importing it** from `../embeddings-deep-dive/embeddings_deep_dive.py`:
`embed_texts` (Gemini batch request), `load_api_key` (key from env or `.env`, never printed) and `cosine_matrix`.
It uses the same model (`gemini-embedding-001`), the same `task_type=SEMANTIC_SIMILARITY`, and the same caching.

Activity 1's cache did not contain these words, so **one new batch API request** was made (2026-11-26 11:56 UTC).
The vectors are cached in `embeddings-cache-90f9c17cdcddeae1.npz`, so later runs make no API call.

To give the score a scale, the script also embeds:
- related words: repayment, compensation;
- same-domain words with a different role: invoice, salary;
- unrelated controls: banana, thunderstorm;
- two example sentences that use the target words in context, plus an unrelated sentence.

Similarity formula: `cosine(A, B) = (A · B) / (|A| × |B|)`. The vectors are unit length here, so this is the dot
product.

```bash
../embeddings-deep-dive/.venv/bin/python word_similarity.py            # uses the cache (no API call)
../embeddings-deep-dive/.venv/bin/python word_similarity.py --refresh  # new API call
```

## Actual results (`results.txt`)

| Measure | Value |
|---|---|
| Model / provider | `gemini-embedding-001`, Google Gemini API |
| Vector dimension | 3072 |
| **cosine(refund, reimbursement)** | **0.9154** (angle about 23.7°) |
| Rank of "reimbursement" among 7 comparison words for "refund" | 1st |
| refund vs repayment / compensation | 0.9085 / 0.8682 |
| refund vs invoice / salary | 0.8440 / 0.8143 |
| refund vs banana / thunderstorm (unrelated) | 0.7820 / 0.7704 |
| Example sentences: refund sentence vs reimbursement sentence | 0.7967 |
| Each of those vs an unrelated sentence (banana breakfast) | 0.7330 / 0.7164 |

An independent recalculation from the cached vectors with a plain Python loop gave 0.915432.

## What the score indicates

- **High and clearly above the baseline.** With this model, even unrelated words score about 0.77. So the useful
  signal is the gap above that baseline: reimbursement scores 0.915, against 0.770–0.782 for the controls.
- **Graded meaning.** Scores fall in order of relatedness: reimbursement (0.915) and repayment (0.909), then
  compensation (0.868), then same-domain words with a different job, invoice and salary (0.81–0.84), then unrelated
  words. Money paid *back* is closer than money owed (invoice) or money earned (salary).
- **Not identical.** Reimbursement is actually slightly closer to compensation (0.929) than to refund (0.915). The
  words overlap in meaning but are not perfect synonyms.
- **Context sentences:** the refund and reimbursement sentences (0.797) are closer to each other than either is to
  the unrelated sentence (0.733 and 0.716). The difference is smaller than for the bare words, because full
  sentences also carry other content (headphones, hotel, taxi).

## Word2Vec versus modern text-embedding models

| | Word2Vec (2013) | Modern text embeddings (e.g. `gemini-embedding-001`) |
|---|---|---|
| Unit embedded | single words, from a fixed vocabulary | any text: a word, sentence or document |
| Context | learned from a small window during training; **one fixed vector per word** | the model reads the whole input, so the same word gets a different representation in different contexts ("river bank" vs "bank deposit") |
| Network | shallow: one hidden layer, the embedding table | deep, transformer-based (built on a large language model) |
| Training signal | predict nearby words (skip-gram or CBOW) | large-scale pre-training, then tuning so that texts with similar meaning get similar vectors |
| Typical size | about 100–300 dimensions | 3072 dimensions here |
| Unknown words | no vector | handled through sub-word pieces |

The core idea is the same in both: meaning is learned from context, and similar meaning becomes nearby vectors.

## Files

| File | Content |
|---|---|
| `word_similarity.py` | Implementation (reuses Activity 1 code) |
| `results.txt` | Actual output of the API run (2026-11-26 11:56 UTC) |
| `embeddings-cache-90f9c17cdcddeae1.npz` | Cached 11×3072 vectors and run metadata (no secrets) |
| `README.md` | This file |
