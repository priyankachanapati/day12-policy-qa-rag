# Day 12 – Activity 5: RAG vs Fine-Tuning

| Item | Detail |
|---|---|
| Module | AI04 - Embeddings, Vector Databases & RAG |
| Type | Conceptual activity: no code and no API calls |
| Files | `Day-12/01-Code/rag-vs-finetuning/` (`README.md`, `decision_analysis.md`) |
| Decisions | HR policy bot: **RAG**; code-style assistant: **fine-tuning**; product catalog Q&A: **RAG** |

## 1. Objective

Explain what RAG adds to an LLM, and when to choose RAG, fine-tuning or both. Apply this to three use cases, with
one line of reasoning each.

## 2. Video details

- **Video:** *RAG vs. Fine Tuning*, IBM Technology (8:57): https://www.youtube.com/watch?v=00Q0G84kq3M
- **Assigned viewing:** full video.
- **Viewing status:** to be watched and confirmed by the learner. This document does not claim the video was
  watched.

## 3. What RAG is

**Retrieval-Augmented Generation (RAG)** gives an LLM an "open-book exam". Before answering, the system looks up
relevant passages in an external knowledge source, such as documents or a database, and pastes them into the prompt.
The model then answers **from those passages**. The model's weights are not changed; the knowledge stays outside the
model, where it can be updated at any time.

**What RAG adds to an LLM:**
- **Fresh and private knowledge** the model never saw in training, such as this week's policies or internal documents.
- **Grounding and citations**, so every answer can point to its source.
- **Fewer made-up facts**, because the model is told to answer only from the retrieved text and to refuse otherwise.
- **Easy updates:** change a document and re-index it; no retraining is needed.

## 4. What fine-tuning is

**Fine-tuning** continues training a pre-trained model on a smaller, task-specific set of examples (for example,
input-and-ideal-output pairs). This **changes the model's weights**, so the new behaviour becomes part of the model.
It can update all weights (full fine-tuning) or only a small set of added weights (parameter-efficient methods such
as LoRA).

Fine-tuning is best at teaching **how to respond**: a style, format, tone, specialised vocabulary or a narrow skill.
It is a poor way to store **facts that change**, because updating them means retraining. The model can also blend or
misremember facts, and it cannot show where an answer came from.

## 5. How RAG works: retrieval, then generation

```
Documents ──► chunk ──► embed ──► vector database          (done once, and again whenever documents change)

Question ──► embed ──► find top-k similar chunks ──► prompt = instructions + chunks + question ──► LLM ──► answer + citations
```

1. **Index:** split the documents into chunks, turn each chunk into an embedding vector, and store the vectors with
   metadata (source, section) in a vector database.
2. **Retrieve:** embed the user's question and fetch the most similar chunks (cosine similarity), optionally filtered
   by metadata.
3. **Generate:** give the LLM the question and the retrieved chunks, with instructions to answer only from them, cite
   them, and refuse if they are not enough.

The Day 12 **Policy Q&A RAG v1** lab is a working example. It retrieves the top-4 policy chunks from ChromaDB, and
`gpt-oss-120b` answers with validated `[Source, Chunk]` citations. It scored 15/15 on the smoke test, including
correct refusals for questions the policies do not cover.

## 6. Key differences

| | RAG | Fine-tuning |
|---|---|---|
| What changes | The **prompt** (retrieved context is added) | The **model weights** |
| Best for | **Knowledge**: facts, documents, data | **Behaviour**: style, format, tone, narrow skills |
| Updating knowledge | Edit and re-index documents (minutes) | Retrain the model (hours or days, plus cost) |
| Freshness | As current as the index | Frozen at training time |
| Traceability | Can cite exact sources | Cannot show where an answer came from |
| Data needed | The documents themselves | Many good-quality training examples |
| Up-front cost | Low (embedding and indexing) | Higher (data preparation, training, evaluation) |
| Per-query cost and latency | Higher (retrieval step and longer prompts) | Lower (short prompts, no retrieval) |
| Access control | Can filter retrieval by user permissions | Everything learned is in the model for every user |
| Main risk | Poor retrieval gives a poor answer | Out-of-date or blended facts, overfitting, forgetting other skills |

## 7. When RAG is preferable

- Answers depend on **facts that change**: policies, prices, stock levels, documentation.
- Users need **citations** or auditability, for example in HR, legal, compliance or support.
- The knowledge is **private** or **too large** to fit in a prompt.
- Different users may see **different documents**, so retrieval must respect permissions.
- Little or no labelled training data is available.

## 8. When fine-tuning is preferable

- You need a **consistent style, format or tone** that is hard to describe fully in a prompt, such as a house coding
  style, a JSON schema or a brand voice.
- A **narrow, repeated task** must be done reliably, such as classification, extraction or domain jargon.
- Prompts are getting very long just to explain the behaviour. Fine-tuning can **bake it in**, which shortens prompts
  and reduces latency and cost per request.
- The required knowledge is **stable** and does not need to be cited.

## 9. When both make sense

Combine them when you need **both a learned behaviour and current facts**. Fine-tune for *how* to answer (tone,
format, domain language, citation habits, refusing correctly) and use RAG for *what* to answer (up-to-date documents).
For example, a support assistant fine-tuned on the company's response style, which retrieves the latest help-centre
articles at question time. In practice, teams usually **start with RAG plus good prompting**, and add fine-tuning
only when evaluation shows a behaviour gap that prompting cannot close.

## 10. The three use cases: decisions and one-line reasoning

| # | Use case | Decision | One-line reasoning |
|---|---|---|---|
| 1 | **HR policy bot** | **RAG** | HR policies change and employees need answers traceable to the current policy, so retrieve and cite the documents rather than freezing them into the weights. |
| 2 | **Code-style assistant** | **Fine-tuning** | Following a team's coding style is a consistent *behaviour* (naming, structure, formatting, idioms) learned best from many examples, not a set of facts to look up. |
| 3 | **Product catalog Q&A** | **RAG** | Catalog facts such as prices, specifications and stock change often and must be exact, so the model should read them from the live catalog at question time. |

Notes on the edge of each decision:
- **HR policy bot.** The Day 12 lab shows this pattern working. Fine-tuning would be added only if the bot also
  needed a specific tone that prompting could not achieve.
- **Code-style assistant.** If the assistant must also follow project-specific APIs or a style guide that changes
  often, add RAG over the style guide and codebase. That makes it **both**, but the core requirement, style, is a
  fine-tuning problem.
- **Product catalog Q&A.** Fine-tuning would bake in prices that are soon wrong. A fine-tuned brand voice could be
  layered on top later, but the facts must come from retrieval.

## 11. Conclusion

RAG and fine-tuning solve different problems. **RAG changes what the model knows at answer time.** It adds current,
private and citable knowledge without retraining, which makes it the right default for policy bots and catalog Q&A.
**Fine-tuning changes how the model behaves.** It bakes in style, format and narrow skills, which suits a code-style
assistant. When an application needs both a learned behaviour and up-to-date facts, the two work well together: a
fine-tuned model that answers from retrieved documents.

## 12. Evidence

This is a conceptual activity. There is no code and no execution log, and no results file was created, because it
would be invented evidence. The analysis is saved in `Day-12/01-Code/rag-vs-finetuning/decision_analysis.md`. The RAG
pattern it recommends is shown working in the Day 12 Policy Q&A RAG v1 lab (smoke test 15/15).
