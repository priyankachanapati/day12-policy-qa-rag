# Day 12 – Activity 1: Embeddings Deep Dive

| Item | Detail |
|---|---|
| Module | AI04 - Embeddings, Vector Databases & RAG |
| Code | `Day-12/01-Code/embeddings-deep-dive/embeddings_deep_dive.py` |
| Embedding model | `gemini-embedding-001` (Google Gemini API), `task_type=SEMANTIC_SIMILARITY`, 3072 dimensions |
| Run | 2026-11-26 11:33 UTC; 1 batch API request for all 20 sentences; all built-in checks PASS |

## 1. Activity objective

Picture embeddings as vectors whose directions encode meaning, and explain why similar meaning gives nearby
vectors. The activity has two parts:
- **Core:** embed 10 sentences, print the cosine-similarity matrix, find the closest pair and explain it.
- **Extension:** embed 20 sentences and show each sentence's nearest neighbours with their scores.

## 2. Embeddings explained in simple language

An **embedding** turns a piece of text into a long list of numbers: here, 3072 numbers per sentence. You can think
of the list as the coordinates of a point, or an arrow, in a space with 3072 directions. The embedding model was
trained so that texts used in similar contexts and carrying similar meaning end up at nearby points. The numbers
themselves are not human-readable. What matters is **where the vectors point relative to each other**.

## 3. How vectors represent meaning

- Each direction in the space loosely corresponds to some aspect of meaning, such as "about weather", "about
  food", "past tense" or "about money". A sentence's vector combines many of these.
- Two sentences about the same thing share many of these aspects, so their arrows point in **similar
  directions**, even when they use different words.
- Words are interpreted **in context**. In "bank of the river", "bank" points towards nature and water; in
  "deposit my salary at the bank" it points towards finance. Our results show this (see Section 12).

## 4. Cosine similarity explained

Cosine similarity measures the **angle** between two vectors, ignoring their length:

`cosine(A, B) = (A · B) / (|A| × |B|)`

- **1.0** means the vectors point the same way (very similar meaning).
- **0** means they are unrelated (at right angles).
- **-1** means they point in opposite directions.

Gemini returned unit-length vectors here (every norm is 1.0000), so cosine similarity is simply the dot product.
With this model, even unrelated sentences score about 0.63-0.75, because all ordinary English sentences share some
"sentence-ness". So **relative** differences matter more than the absolute value.

## 5. Video reference and assigned timestamps

- **Video:** *Transformers, the tech behind LLMs | Deep Learning Chapter 5*, 3Blue1Brown:
  https://www.youtube.com/watch?v=wjZofJX0v4M
- **Assigned viewing:** 12:27-20:22 ("Word embeddings" and "Embeddings beyond words").
- **Viewing status:** to be completed and confirmed by the learner. This document does not claim the video was
  watched.

## 6. Python implementation

| Step | How it is done |
|---|---|
| Dataset | 20 sentences with IDs S01-S20 and a theme label, stored in the script |
| Embedding | One REST call to `models/gemini-embedding-001:batchEmbedContents`. The key is sent in a header, never in the URL or output. Falls back to one request per sentence if batching is unavailable |
| Errors | Clear messages for a missing key, HTTP errors (429/5xx retried with backoff) and network errors |
| Caching | Vectors saved to `results/embeddings-cache-*.npz`, so re-runs make no API calls (`--refresh` forces a new call) |
| Similarity | Vectors L2-normalised, then the matrix product gives every cosine value |
| Results | 10x10 core matrix and closest pair; top-3 nearest neighbours for all 20; theme statistics; CSV and JSON files |
| Verification | Built-in PASS/FAIL checks (count, dimension, finite values, 20x20 shape, symmetry, diagonal = 1, range [-1, 1], sorted neighbours) |

Key code:

```python
def cosine_matrix(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    unit = vectors / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)

def closest_pair(sim):
    masked = sim.copy()
    np.fill_diagonal(masked, -np.inf)        # ignore self-similarity
    i, j = np.unravel_index(np.argmax(masked), masked.shape)
    return i, j, sim[i, j]
```

## 7. Dataset of 20 sentences

S01-S10 are the **core activity**; S11-S20 extend it to 20. The set deliberately includes paraphrases that share
few words (S01/S02, S03/S04, S06/S14, S07/S16, S08/S17) and a word-overlap trap (S19/S20, "bank").

| ID | Theme | Sentence |
|---|---|---|
| S01 | weather | It is raining heavily outside today. |
| S02 | weather | There is a lot of rain falling this afternoon. |
| S03 | food | I cooked pasta with tomato sauce for dinner. |
| S04 | food | We had spaghetti in a red sauce for our evening meal. |
| S05 | sports | The football team won the championship final. |
| S06 | technology | My laptop keeps crashing when I open too many programs. |
| S07 | travel | We booked a flight to Tokyo for our summer holiday. |
| S08 | learning | She is studying hard for her mathematics exam. |
| S09 | sports | The striker scored the winning goal in the last minute. |
| S10 | technology | The new phone has a faster processor and a better camera. |
| S11 | weather | The forecast says it will be sunny and warm tomorrow. |
| S12 | food | Fresh fruit and vegetables are part of a healthy diet. |
| S13 | sports | Tennis players need quick reflexes and good stamina. |
| S14 | technology | My computer froze and I had to restart it. |
| S15 | travel | The train journey through the mountains was beautiful. |
| S16 | travel | Our holiday flight to Japan departs next week. |
| S17 | learning | He is revising for his maths test tomorrow. |
| S18 | learning | Online courses make it easier to learn new skills. |
| S19 | nature | The bank of the river was flooded after the storm. |
| S20 | finance | I need to go to the bank to deposit my salary. |

## 8. Cosine-similarity matrix for the first 10 sentences

Actual output (`results/similarity-matrix-core-10.csv` has 6-decimal values):

```
         S01    S02    S03    S04    S05    S06    S07    S08    S09    S10
S01    1.000  0.942  0.718  0.734  0.714  0.732  0.697  0.727  0.709  0.703
S02    0.942  1.000  0.708  0.731  0.720  0.743  0.696  0.721  0.704  0.713
S03    0.718  0.708  1.000  0.900  0.668  0.683  0.705  0.682  0.689  0.679
S04    0.734  0.731  0.900  1.000  0.704  0.674  0.733  0.683  0.698  0.689
S05    0.714  0.720  0.668  0.704  1.000  0.632  0.698  0.688  0.829  0.707
S06    0.732  0.743  0.683  0.674  0.632  1.000  0.663  0.718  0.664  0.735
S07    0.697  0.696  0.705  0.733  0.698  0.663  1.000  0.686  0.675  0.704
S08    0.727  0.721  0.682  0.683  0.688  0.718  0.686  1.000  0.697  0.712
S09    0.709  0.704  0.689  0.698  0.829  0.664  0.675  0.697  1.000  0.708
S10    0.703  0.713  0.679  0.689  0.707  0.735  0.704  0.712  0.708  1.000
```

The three highest off-diagonal values in this matrix are the three same-theme pairs: S01/S02 (0.942), S03/S04
(0.900) and S05/S09 (0.829). The least similar pair is S05/S06, football versus a crashing laptop (0.632).

## 9. Closest pair and explanation

**Closest pair: S01 & S02, cosine similarity 0.9419.** It is the closest pair in both the core set and all 20.

- S01: *It is raining heavily outside today.*
- S02: *There is a lot of rain falling this afternoon.*

**Why their embeddings are similar:** both sentences describe the same situation, heavy rain happening now. Their
only shared content word is a form of "rain" ("raining" and "rain"). The rest is worded differently: "heavily
outside today" versus "a lot of ... falling this afternoon". The model maps "heavily" and "a lot of", and "today"
and "this afternoon", to similar meanings. So the two vectors point in almost the same direction, at an angle of
about 20° (arccos 0.942).

By contrast, S03/S04 (pasta vs spaghetti, tomato vs red sauce, dinner vs evening meal) share almost no words, yet
still score 0.900. That shows the similarity comes from **meaning, not word overlap**.

## 10. Nearest neighbours for the 20 sentences

Top-3 neighbours for every sentence (actual output):

| Sentence | 1st neighbour | 2nd | 3rd | Nearest shares theme? |
|---|---|---|---|---|
| S01 weather | S02 0.942 | S19 0.815 | S11 0.759 | yes |
| S02 weather | S01 0.942 | S19 0.810 | S11 0.772 | yes |
| S03 food | S04 0.900 | S14 0.743 | S12 0.742 | yes |
| S04 food | S03 0.900 | S12 0.751 | S01 0.734 | yes |
| S05 sports | S09 0.829 | S15 0.738 | S13 0.726 | yes |
| S06 technology | S14 0.845 | S02 0.743 | S10 0.735 | yes |
| S07 travel | S16 0.888 | S15 0.734 | S04 0.733 | yes |
| S08 learning | S17 0.878 | S01 0.727 | S13 0.726 | yes |
| S09 sports | S05 0.829 | S13 0.738 | S17 0.711 | yes |
| S10 technology | S18 0.764 | S13 0.752 | S14 0.741 | no |
| S11 weather | S02 0.772 | S01 0.759 | S16 0.745 | yes |
| S12 food | S13 0.773 | S04 0.751 | S03 0.742 | no |
| S13 sports | S12 0.773 | S10 0.752 | S09 0.738 | no |
| S14 technology | S06 0.845 | S19 0.749 | S03 0.743 | yes |
| S15 travel | S05 0.738 | S07 0.734 | S02 0.720 | no |
| S16 travel | S07 0.888 | S11 0.745 | S20 0.744 | yes |
| S17 learning | S08 0.878 | S16 0.732 | S13 0.731 | yes |
| S18 learning | S10 0.764 | S13 0.737 | S12 0.733 | no |
| S19 nature | S01 0.815 | S02 0.810 | S14 0.749 | no (only sentence in its theme) |
| S20 finance | S19 0.746 | S16 0.744 | S14 0.732 | no (only sentence in its theme) |

**Why similar meanings score higher:** every paraphrase pair found its partner as nearest neighbour, with the
highest scores in the table:
- S07 <-> S16 (Tokyo flight / Japan flight): 0.888
- S08 <-> S17 (maths exam / maths test): 0.878
- S06 <-> S14 (laptop crashing / computer froze): 0.845

The model places sentences about the same situation in nearly the same direction, regardless of vocabulary.

## 11. Actual execution output

The full output is in `Day-12/01-Code/embeddings-deep-dive/results/execution-output.txt` (92 lines). The header and
checks from that run:

```
Embedding model : gemini-embedding-001 (Google Gemini API), task_type=SEMANTIC_SIMILARITY
Embeddings from : Gemini API (batchEmbedContents (1 request)); cached to embeddings-cache-bd98ab393b252255.npz
API key source  : GEMINI_API_KEY from policy-qa-rag/.env (value never printed)
Sentences       : 20 | embedding dimension: 3072
Verification checks:
  [PASS] 20 sentences embedded
  [PASS] all embeddings same dimension
  [PASS] no NaN/inf values
  [PASS] similarity matrix is 20x20
  [PASS] matrix symmetric
  [PASS] diagonal = 1.0 (self-similarity)
  [PASS] all scores within [-1, 1]
  vector L2 norms: min=1.0000 max=1.0000
...
  [PASS] every neighbour list sorted by descending similarity
Closest pair overall (S01-S20): S01 & S02  cosine = 0.9419
Observations computed from the matrix:
  mean cosine, same-theme pairs      : 0.7854  (18 pairs)
  mean cosine, different-theme pairs : 0.7066  (172 pairs)
  nearest neighbour shares the theme : 13/20 sentences
  'bank' trap S19 (river) vs S20 (money): cosine = 0.7465; S19's nearest = S01, S20's nearest = S19
```

An independent check, which recomputed every cosine value with a plain Python loop from the saved vectors, matched
the saved matrix to within 5×10⁻⁷ and confirmed the closest pair and the neighbour ordering.

## 12. Key observations and learning outcomes

1. **Meaning beats word overlap.**
   - Paraphrases with few shared words scored 0.845-0.942.
   - The two "bank" sentences share a key word but scored only 0.747.
   - S19 (river bank flooded after a storm) is closer to the **rain** sentences S01 (0.815) and S02 (0.810) than to
     the money "bank" sentence S20.
2. **Themes form neighbourhoods.** Same-theme pairs average 0.785, against 0.707 for different-theme pairs. For
   13 of the 18 sentences whose theme has at least one other sentence, the nearest neighbour is from the same theme.
3. **Neighbours follow meaning, not our labels.** Some nearest neighbours cross theme labels in sensible ways:
   - S10 (new phone with faster processor) -> S18 (online courses, learning new skills): 0.764
   - S12 (healthy diet) -> S13 (tennis players' stamina), both about health and fitness: 0.773
   - S15 (scenic train journey) -> S05 (football final): 0.738. Its weakest top match, which suggests no other
     sentence is strongly about scenic travel.
4. **Read scores relatively.** With this model, unrelated sentences still score about 0.63-0.75. A score means more
   when compared with the other scores for the same sentence than as an absolute number.
5. **Link to the RAG lab.** This is the same principle the Policy Q&A RAG lab uses to retrieve relevant policy
   chunks. A question and the chunk that answers it point in similar directions, so cosine similarity finds that
   chunk even when the wording differs.

## 13. Evidence file references

| Evidence | File | What it shows |
|---|---|---|
| Source code | `01-Code/embeddings-deep-dive/embeddings_deep_dive.py` | Dataset, API call, cosine maths, checks |
| README | `01-Code/embeddings-deep-dive/README.md` | Setup and run instructions |
| Execution output | `01-Code/embeddings-deep-dive/results/execution-output.txt` | Full output of the API run |
| Core matrix | `01-Code/embeddings-deep-dive/results/similarity-matrix-core-10.csv` | 10x10 cosine values |
| Full matrix | `01-Code/embeddings-deep-dive/results/similarity-matrix-all-20.csv` | 20x20 cosine values |
| Nearest neighbours | `01-Code/embeddings-deep-dive/results/nearest-neighbours-20.json` | Top-3 neighbours with scores |
| Screenshot | `03-Evidence/10-embeddings-matrix-core-10.png` | Header, checks, sentences, 10x10 matrix, closest pair |
| Screenshot | `03-Evidence/11-embeddings-nearest-neighbours-20.png` | Nearest neighbours for 20 sentences and observations |
