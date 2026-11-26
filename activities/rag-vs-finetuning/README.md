# Day 12 – Activity 5: RAG vs Fine-Tuning

Module AI04 (Embeddings, Vector Databases & RAG). **Objective:** explain what RAG adds to an LLM and when to choose
RAG versus fine-tuning.

**Learning reference (for the learner to watch; viewing is not claimed here):** *RAG vs. Fine Tuning*, IBM Technology,
8:57, https://www.youtube.com/watch?v=00Q0G84kq3M (full video).

## Practical activity: the three decisions

| Use case | Decision | One-line reasoning |
|---|---|---|
| HR policy bot | **RAG** | HR policies change and employees need answers traceable to the current policy, so retrieve and cite the documents rather than freezing them into the weights. |
| Code-style assistant | **Fine-tuning** | Following a team's coding style is a consistent *behaviour* (naming, structure, formatting, idioms) learned best from many examples, not a set of facts to look up. |
| Product catalog Q&A | **RAG** | Catalog facts such as prices, specifications and stock change often and must be exact, so the model should read them from the live catalog at question time. |

The full explanation is in [`decision_analysis.md`](decision_analysis.md). It covers what RAG and fine-tuning are, how
RAG works, a comparison table, when to prefer each, when to combine them, edge cases for each decision, and a
conclusion.

## Execution evidence

This is a **conceptual activity**. No code was written, no API calls were made, and there is therefore **no
`results.txt`**: an execution log would be invented evidence. The RAG pattern discussed here is shown working in
the Day 12 **Policy Q&A RAG v1** lab (`../policy-qa-rag/`, smoke test 15/15).

## Files

| File | Content |
|---|---|
| `README.md` | Overview and the three decisions |
| `decision_analysis.md` | Full analysis (sections 1–9) |
