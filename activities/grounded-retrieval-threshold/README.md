# Day 12 – Activity 7: Grounded Retrieval and Relevance Threshold

Module AI04 (Embeddings, Vector Databases & RAG). **Objective:** query a vector store for relevant chunks and write a
grounded prompt that produces a cited answer. Add a relevance-score threshold so the bot says **"I don't know"** when
nothing scores high enough.

**Learning reference (for the learner to watch; viewing is not claimed here):** *RAG + Langchain Python Project:
Easy AI/Chat For Your Docs*, pixegami, https://www.youtube.com/watch?v=tcqEUSNCn8I, 09:38–16:18 (chapters: Querying
for Relevant Data, Crafting a Great Response).

## Reuse

`retriever.py` **imports Activity 6** (`../rag-from-scratch/rag.py`) and reuses:
- its Chroma database (`rag_db/`, 60 chunks from 20 policy documents);
- the local all-MiniLM-L6-v2 embeddings;
- `retrieve()`, the grounded system prompt, the Groq call (`openai/gpt-oss-120b`) and the citation validator.

No new database or embedding model is created.

## Score metric and threshold logic

| Score | Range | Meaning |
|---|---|---|
| **Cosine distance** `d` (what Chroma returns) | 0 to 2 | **Lower = more relevant** |
| **Cosine similarity** `s = 1 - d` (what the threshold uses) | -1 to 1 | **Higher = more relevant** |

**Rule:** answer only if the **best** retrieved chunk has **similarity s ≥ T**, which is the same as distance
d ≤ 1 − T. Otherwise reply *"I don't know. The policy documents do not contain information relevant to this
question."* **without calling the LLM**. When the rule passes, only chunks with s ≥ T are put in the prompt.

**Choosing T (calibration, kept separate from testing).** T was chosen from 12 calibration questions that are
*not* reused as test questions:
- 6 in scope: the lowest best-similarity was **0.5731** (hotel limit).
- 6 off topic: the highest was **0.3106** (pets in the office, a near-topic distractor). Plain off-topic questions
  scored 0.09–0.16.
- **T = midpoint, rounded = 0.44**, which is a distance cut-off of 0.56.

## Actual results (`results.txt`, 2026-11-26 13:02 UTC)

| Test | Question | Best similarity (distance) | Threshold decision | Final response |
|---|---|---|---|---|
| T1 in-scope | When is the IT Service Desk available? | **0.7203** (0.2797) | PASS → retrieve + generate | "The IT Service Desk is available from 08:00 to 18:00 on working days. [it-03-service-desk::contact::0]" |
| T2 in-scope | How do I report a phishing email? | **0.6651** (0.3349) | PASS → retrieve + generate | "Suspicious emails should be reported using the Report Phishing button [sec-03-phishing-and-incidents::reporting::0]." |
| T3 off-topic | What is a good recipe for chocolate cake? | **0.0559** (0.9441) | BELOW 0.44 → LLM not called | "I don't know. The policy documents do not contain information relevant to this question." |
| T4 off-topic | Who won the football World Cup in 2018? | **0.1191** (0.8809) | BELOW 0.44 → LLM not called | "I don't know. The policy documents do not contain information relevant to this question." |

- **Citation check:** both in-scope answers cite exactly one chunk, and it is one of the chunks sent to the LLM.
  There were 0 invalid citations.
- **LLM calls:** 2, only for the questions that passed the threshold. **OVERALL: PASS.**
- **Offline tests (`offline-tests.txt`, also appended to `results.txt`): 8 passed.** They check:
  - similarity = 1 − distance, and best-first ordering;
  - the calibration separates the two sets;
  - the test questions are not calibration questions;
  - both off-topic questions return "I don't know" without the LLM;
  - both in-scope questions pass, and only chunks with s ≥ T reach the LLM;
  - a citation to a chunk that was not retrieved is withheld.

**Observations and limitations:**
- The gap between the two groups is wide: the off-topic tests scored 0.06–0.12, against 0.67–0.72 in scope.
- The threshold is specific to this embedding model and this document set. A different model or corpus needs
  re-calibration.
- A borderline but unrelated chunk can still clear T. In T1, `work-02-office-guidelines::desk-booking::0` (0.4921)
  was sent along with the Service Desk chunks. The grounded prompt and citation check kept the answer correct.
- The threshold is one safety layer. The prompt's NOT_FOUND rule still handles in-scope-looking questions whose
  answer is missing, as with the "pets" question in Activity 6.

## Run

```bash
# uses Activity 6's environment and database
../rag-from-scratch/.venv/bin/python retriever.py          # calibration + 4 tests -> results.txt
../rag-from-scratch/.venv/bin/python -m pytest -v          # 8 offline tests (no API)
```

## Files

| File | Content |
|---|---|
| `retriever.py` | Threshold calibration, `retrieve_with_threshold()`, `grounded_answer()`, demo |
| `test_threshold.py` | 8 offline tests |
| `results.txt` | Actual live output, plus the offline test output |
| `offline-tests.txt` | Actual pytest output |
