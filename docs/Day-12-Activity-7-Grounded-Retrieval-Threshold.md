# Day 12 – Activity 7: Grounded Retrieval and Relevance Threshold

| Item | Detail |
|---|---|
| Module | AI04 - Embeddings, Vector Databases & RAG |
| Code | `Day-12/01-Code/grounded-retrieval-threshold/retriever.py` (reuses Activity 6 `rag.py`) |
| Score metric | cosine **similarity** s = 1 - cosine distance (higher = more relevant) |
| Threshold | **T = 0.44** on similarity (distance <= 0.56), calibrated on a separate question set |
| Result | 2 in-scope questions answered with valid citations; 2 off-topic questions answered "I don't know" with no LLM call; OVERALL PASS; 8 offline tests passed |

## 1. Objective

Query a vector store for relevant chunks and use a grounded prompt to produce a cited answer. Add a relevance-score
threshold so the bot says "I don't know" when nothing scores high enough, and test it with two off-topic questions.

## 2. Video details

- **Video:** *RAG + Langchain Python Project: Easy AI/Chat For Your Docs*, pixegami:
  https://www.youtube.com/watch?v=tcqEUSNCn8I
- **Assigned viewing:** 09:38-16:18 (chapters: Querying for Relevant Data, Crafting a Great Response).
- **Viewing status:** to be watched and confirmed by the learner. This document does not claim the video was
  watched.

## 3. What grounded retrieval means

A *grounded* answer is built only from retrieved source text, and cites where each fact came from. Retrieval finds
the closest chunks. A strict prompt then tells the LLM to use only those chunks, cite their IDs, and say it does not
know otherwise.

Retrieval always returns *something*, even for a nonsense question. Without a **relevance threshold**, weak and
unrelated chunks would still be sent to the LLM, inviting a made-up answer. The threshold stops that before
generation: if even the best chunk is not similar enough, the bot replies "I don't know" and the LLM is never
called. That also saves API cost.

## 4. Reuse

`retriever.py` **imports Activity 6** (`../rag-from-scratch/rag.py`) and reuses:
- its Chroma database (`rag_db/`, 60 chunks from 20 policy documents);
- the local all-MiniLM-L6-v2 embeddings;
- `retrieve()`, the grounded system prompt, the Groq call (`openai/gpt-oss-120b`) and the citation validator.

No new database or embedding model is created.

## 5. Score interpretation and threshold logic

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

Code:

```python
def retrieve_with_threshold(query: str, threshold: float, k: int = K) -> dict:
    """Retrieve top-k chunks and apply the threshold to the BEST chunk's similarity."""
    best_sim, best_dist, hits = best_similarity(query, k)
    passed = best_sim >= threshold
    return {
        "query": query,
        "hits": hits,
        "best_similarity": best_sim,
        "best_distance": best_dist,
        "threshold": threshold,
        "passed": passed,
        "context": [h for h in hits if h["similarity"] >= threshold],  # only chunks above T go to the LLM
    }
```

## 6. Sample in-scope questions, off-topic questions, scores and final responses

| Test | Question | Best similarity (distance) | Threshold decision | Final response |
|---|---|---|---|---|
| T1 in-scope | When is the IT Service Desk available? | **0.7203** (0.2797) | PASS → retrieve + generate | "The IT Service Desk is available from 08:00 to 18:00 on working days. [it-03-service-desk::contact::0]" |
| T2 in-scope | How do I report a phishing email? | **0.6651** (0.3349) | PASS → retrieve + generate | "Suspicious emails should be reported using the Report Phishing button [sec-03-phishing-and-incidents::reporting::0]." |
| T3 off-topic | What is a good recipe for chocolate cake? | **0.0559** (0.9441) | BELOW 0.44 → LLM not called | "I don't know. The policy documents do not contain information relevant to this question." |
| T4 off-topic | Who won the football World Cup in 2018? | **0.1191** (0.8809) | BELOW 0.44 → LLM not called | "I don't know. The policy documents do not contain information relevant to this question." |

## 7. Citation verification and tests

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

## 8. Actual output (results.txt, live run, verbatim)

```
================================================================================================
DAY 12 - ACTIVITY 7: GROUNDED RETRIEVAL WITH A RELEVANCE THRESHOLD - 2026-11-26T13:02:36+00:00
================================================================================================
Reused from Activity 6: rag.retrieve (Chroma rag_db/, 60 chunks, all-MiniLM-L6-v2 local), grounded prompt,
Groq call (openai/gpt-oss-120b) and citation validator.
Score metric: Chroma cosine DISTANCE d (lower = closer) -> cosine SIMILARITY s = 1 - d (higher = closer).
Rule: answer only if the BEST chunk has s >= T (i.e. d <= 1 - T); otherwise reply "I don't know" (no LLM call).

STEP 1 - CALIBRATION (separate questions, used only to choose T)
  0.8139  in-scope   How many days of paid annual leave do full-time employees get?
  0.7669  in-scope   What should I do if I lose the phone with my authenticator app?
  0.7518  in-scope   How often are fire drills held?
  0.6990  in-scope   How long must my password be?
  0.6343  in-scope   How many days per week can I work from home?
  0.5731  in-scope   What is the hotel limit per night on business travel?
  0.3106  off-topic  What is the company's policy on bringing pets to the office?
  0.1569  off-topic  What is the current share price of Apple?
  0.1536  off-topic  How do I learn to play the guitar?
  0.1136  off-topic  What will the weather be like tomorrow?
  0.0981  off-topic  What is the capital of Australia?
  0.0897  off-topic  Which planet is closest to the sun?
  lowest in-scope = 0.5731 | highest off-topic = 0.3106 | T = midpoint rounded = 0.44  (distance cut-off = 0.56)

STEP 2 - TEST QUESTIONS (not used for calibration)

T1 (in-scope): When is the IT Service Desk available?
  retrieved: it-03-service-desk::contact::0 (s=0.7203, d=0.2797), it-03-service-desk::out-of-hours::0 (s=0.5535, d=0.4465), it-03-service-desk::ticket-priorities::0 (s=0.5161, d=0.4839), work-02-office-guidelines::desk-booking::0 (s=0.4921, d=0.5079)
  best similarity = 0.7203 (distance 0.2797) vs T = 0.44 -> PASS: retrieve + generate
  chunks sent to LLM (s >= T): ['it-03-service-desk::contact::0', 'it-03-service-desk::out-of-hours::0', 'it-03-service-desk::ticket-priorities::0', 'work-02-office-guidelines::desk-booking::0']
  status: answered | LLM called: True
  final answer: The IT Service Desk is available from 08:00 to 18:00 on working days. [it-03-service-desk::contact::0]
  citations: ['it-03-service-desk::contact::0'] | invalid: none
  [PASS] grounded answer with citations that are all retrieved chunks

T2 (in-scope): How do I report a phishing email?
  retrieved: sec-03-phishing-and-incidents::reporting::0 (s=0.6651, d=0.3349), sec-03-phishing-and-incidents::recognising-phishing::0 (s=0.5217, d=0.4783), sec-03-phishing-and-incidents::incident-response::0 (s=0.4611, d=0.5389), hr-01-code-of-conduct::reporting-concerns::0 (s=0.3225, d=0.6775)
  best similarity = 0.6651 (distance 0.3349) vs T = 0.44 -> PASS: retrieve + generate
  chunks sent to LLM (s >= T): ['sec-03-phishing-and-incidents::reporting::0', 'sec-03-phishing-and-incidents::recognising-phishing::0', 'sec-03-phishing-and-incidents::incident-response::0']
  status: answered | LLM called: True
  final answer: Suspicious emails should be reported using the Report Phishing button [sec-03-phishing-and-incidents::reporting::0].
  citations: ['sec-03-phishing-and-incidents::reporting::0'] | invalid: none
  [PASS] grounded answer with citations that are all retrieved chunks

T3 (off-topic): What is a good recipe for chocolate cake?
  retrieved: work-04-travel-and-expenses::meals::0 (s=0.0559, d=0.9441), work-01-remote-work::core-hours::0 (s=0.0490, d=0.9510), leave-02-sick-leave::notification::0 (s=0.0449, d=0.9551), work-04-travel-and-expenses::booking-travel::0 (s=0.0382, d=0.9618)
  best similarity = 0.0559 (distance 0.9441) vs T = 0.44 -> BELOW THRESHOLD: I dont know (LLM not called)
  status: i_dont_know | LLM called: False
  final answer: I don't know. The policy documents do not contain information relevant to this question.
  [PASS] off-topic question answered "I don't know" without calling the LLM

T4 (off-topic): Who won the football World Cup in 2018?
  retrieved: hr-02-performance-reviews::review-cycle::0 (s=0.1191, d=0.8809), it-02-software-installation::unapproved-software::0 (s=0.1061, d=0.8939), it-01-laptop-and-equipment::issued-equipment::0 (s=0.0857, d=0.9143), it-02-software-installation::requesting-new-software::0 (s=0.0682, d=0.9318)
  best similarity = 0.1191 (distance 0.8809) vs T = 0.44 -> BELOW THRESHOLD: I dont know (LLM not called)
  status: i_dont_know | LLM called: False
  final answer: I don't know. The policy documents do not contain information relevant to this question.
  [PASS] off-topic question answered "I don't know" without calling the LLM

LLM calls made: 2 (only questions that passed the threshold)
OVERALL: PASS
```

## 9. Code location and evidence

| Item | Location |
|---|---|
| Threshold retriever and demo | `Day-12/01-Code/grounded-retrieval-threshold/retriever.py` |
| Offline tests | `Day-12/01-Code/grounded-retrieval-threshold/test_threshold.py`, `offline-tests.txt` |
| Actual output | `Day-12/01-Code/grounded-retrieval-threshold/results.txt` |
| Reused pipeline | `Day-12/01-Code/rag-from-scratch/rag.py` and `rag_db/` |
| README | `Day-12/01-Code/grounded-retrieval-threshold/README.md` |
