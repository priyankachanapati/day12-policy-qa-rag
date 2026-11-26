# Day 12 – Activity 2: Word2Vec and How Embeddings Are Learned

| Item | Detail |
|---|---|
| Module | AI04 - Embeddings, Vector Databases & RAG |
| Code | `Day-12/01-Code/word2vec-contextual-embeddings/word_similarity.py` (reuses Activity 1 code) |
| Model | `gemini-embedding-001` (Google Gemini API), `task_type=SEMANTIC_SIMILARITY`, 3072 dimensions |
| Run | 2026-11-26 11:56 UTC; 1 batch API request; all built-in checks PASS |

## 1. Learning objective

Explain how a neural network learns embeddings from context, and check with the similarity script why "refund" and
"reimbursement" end up close in embedding space.

## 2. Video details

- **Video:** *Word Embedding and Word2Vec, Clearly Explained!!!*, StatQuest with Josh Starmer:
  https://www.youtube.com/watch?v=viZrOnJclY0
- **Assigned viewing:** 00:00-11:44 (about 12 minutes).
- **Viewing status:** to be watched and confirmed by the learner. This document does not claim the video was
  watched.

## 3. The answer in exactly three sentences

"Refund" and "reimbursement" appear in very similar contexts, surrounded by words such as money, paid, back,
receipt, claim and expense, even though a refund usually follows a returned purchase and a reimbursement usually
covers a cost someone paid on another's behalf. An embedding model is trained to predict words from their
surroundings, so words that keep the same company are nudged toward the same region of vector space. Our similarity
script confirms this: the cosine similarity between the two words is 0.9154, the highest of the seven comparison
words tested against "refund".

## 4. How a neural network learns embeddings from context

1. **Start random.** Every word in the vocabulary gets a vector of random numbers.
2. **Make a prediction task from plain text.** Word2Vec slides a small window over sentences. In the *skip-gram*
   version, the network sees one word (for example "refund") and must predict its neighbours ("issued", "full",
   "receipt"). In *CBOW* it does the reverse, predicting the middle word from its neighbours.
3. **Learn from mistakes.** Each wrong prediction produces an error. Backpropagation then adjusts the word's
   vector, and the output weights, slightly in the direction that would have made the right neighbours more likely.
4. **Repeat millions of times.** Words that keep predicting the same neighbours receive similar updates again and
   again, so their vectors drift toward each other. The vector table *is* the embedding. Nobody labels meanings; the
   meaning comes from the context ("you shall know a word by the company it keeps").

## 5. Example sentences

The words in context, as embedded by the script:
- **Refund:** "The store gave me a refund after I returned the broken headphones."
- **Reimbursement:** "The company paid my reimbursement for the hotel and taxi costs."
- **Unrelated control:** "I ate a banana for breakfast this morning."

Both target sentences describe money coming back to a person. That shared idea is what the model captures.

## 6. Similarity formula

`cosine(A, B) = (A · B) / (|A| × |B|)`

- 1.0 means the vectors point in the same direction.
- 0 means they are unrelated (at right angles).
- -1 means they point in opposite directions.

The Gemini vectors are unit length (norm 1.0000), so the cosine is the dot product.

## 7. Actual results

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

## 8. Interpretation of the score

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

## 9. Word2Vec versus modern text-embedding models

| | Word2Vec (2013) | Modern text embeddings (e.g. `gemini-embedding-001`) |
|---|---|---|
| Unit embedded | single words, from a fixed vocabulary | any text: a word, sentence or document |
| Context | learned from a small window during training; **one fixed vector per word** | the model reads the whole input, so the same word gets a different representation in different contexts ("river bank" vs "bank deposit") |
| Network | shallow: one hidden layer, the embedding table | deep, transformer-based (built on a large language model) |
| Training signal | predict nearby words (skip-gram or CBOW) | large-scale pre-training, then tuning so that texts with similar meaning get similar vectors |
| Typical size | about 100–300 dimensions | 3072 dimensions here |
| Unknown words | no vector | handled through sub-word pieces |

The core idea is the same in both: meaning is learned from context, and similar meaning becomes nearby vectors.

## 10. Code location and evidence

| Item | Location |
|---|---|
| Implementation | `Day-12/01-Code/word2vec-contextual-embeddings/word_similarity.py` |
| Reused code | `Day-12/01-Code/embeddings-deep-dive/embeddings_deep_dive.py` (`embed_texts`, `load_api_key`, `cosine_matrix`) |
| Actual output | `Day-12/01-Code/word2vec-contextual-embeddings/results.txt` |
| Cached vectors | `Day-12/01-Code/word2vec-contextual-embeddings/embeddings-cache-90f9c17cdcddeae1.npz` |
| README | `Day-12/01-Code/word2vec-contextual-embeddings/README.md` |

Actual output (`results.txt`, verbatim):

```
============================================================================================
DAY 12 - ACTIVITY 2: WORD2VEC AND HOW EMBEDDINGS ARE LEARNED - refund vs reimbursement
============================================================================================
Embedding model : gemini-embedding-001 (Google Gemini API), task_type=SEMANTIC_SIMILARITY
Embeddings from : Gemini API (batchEmbedContents (1 request)); cached to embeddings-cache-90f9c17cdcddeae1.npz
API key source  : GEMINI_API_KEY from ../policy-qa-rag/.env (value never printed)
Items embedded  : 11 | vector dimension: 3072
Code reused from: ../embeddings-deep-dive/embeddings_deep_dive.py (embed_texts, cosine_matrix, load_api_key)

Checks: [PASS] 11 vectors returned | [PASS] same dimension for all | [PASS] finite values | [PASS] scores within [-1, 1] | [PASS] self-similarity = 1.0

RESULT  cosine(refund, reimbursement) = 0.9154   (angle ~ 23.7 degrees)

Word comparisons (cosine similarity to each target word):
  word            role                                      vs refund  vs reimbursement
  repayment       related: money paid back                     0.9085            0.9102
  compensation    related: money for a loss                    0.8682            0.9289
  invoice         same domain, different role (a bill)         0.8440            0.8646
  salary          same domain, different role (wages)          0.8143            0.8569
  banana          unrelated control                            0.7820            0.7839
  thunderstorm    unrelated control                            0.7704            0.7671
  Ranking of all words by similarity to 'refund': reimbursement 0.915, repayment 0.908, compensation 0.868, invoice 0.844, salary 0.814, banana 0.782, thunderstorm 0.770
  'reimbursement' rank among 7 comparison words for 'refund': 1

Example sentences (words used in context):
  ctx_refund  : The store gave me a refund after I returned the broken headphones.
  ctx_reimb   : The company paid my reimbursement for the hotel and taxi costs.
  ctx_control : I ate a banana for breakfast this morning.
  cosine(ctx_refund, ctx_reimb)   = 0.7967
  cosine(ctx_refund, ctx_control) = 0.7330
  cosine(ctx_reimb, ctx_control)  = 0.7164
```
