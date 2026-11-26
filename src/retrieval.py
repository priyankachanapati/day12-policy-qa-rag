"""Top-k semantic retrieval.

Scoring convention (used everywhere in this project):

* ``distance``   - cosine distance from ChromaDB, range 0..2, **lower is better**.
* ``similarity`` - ``1 - distance`` = cosine similarity, range -1..1, **higher is better**.

Chunks whose similarity is below ``min_similarity`` are treated as not relevant.
If no chunk clears the threshold, the pipeline refuses without calling the LLM.
"""

from __future__ import annotations

from dataclasses import dataclass

from .embeddings import LocalEmbedder
from .vector_store import PolicyVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A retrieved chunk with its scores and metadata."""

    rank: int
    chunk_id: str
    policy_id: str
    title: str
    section: str
    source_file: str
    text: str
    distance: float
    similarity: float


@dataclass(frozen=True)
class RetrievalResult:
    """All top-k results for a question, plus the subset above the relevance threshold."""

    question: str
    top_k: int
    min_similarity: float
    chunks: list[RetrievedChunk]

    @property
    def relevant_chunks(self) -> list[RetrievedChunk]:
        return [c for c in self.chunks if c.similarity >= self.min_similarity]

    @property
    def has_relevant_context(self) -> bool:
        return bool(self.relevant_chunks)


class Retriever:
    """Embeds a question and returns the top-k most similar chunks."""

    def __init__(
        self,
        embedder: LocalEmbedder,
        store: PolicyVectorStore,
        top_k: int = 4,
        min_similarity: float = 0.25,
    ) -> None:
        self.embedder = embedder
        self.store = store
        self.top_k = top_k
        self.min_similarity = min_similarity

    def retrieve(self, question: str, top_k: int | None = None) -> RetrievalResult:
        """Retrieve the top-k chunks for ``question``."""
        if not question or not question.strip():
            raise ValueError("Question must not be empty")
        k = top_k or self.top_k
        matches = self.store.query(self.embedder.embed_one(question.strip()), k)
        chunks = [
            RetrievedChunk(
                rank=i,
                chunk_id=m.chunk_id,
                policy_id=str(m.metadata.get("policy_id", "")),
                title=str(m.metadata.get("title", "")),
                section=str(m.metadata.get("section", "")),
                source_file=str(m.metadata.get("source_file", "")),
                text=m.text,
                distance=m.distance,
                similarity=1.0 - m.distance,
            )
            for i, m in enumerate(matches, start=1)
        ]
        return RetrievalResult(
            question=question.strip(), top_k=k, min_similarity=self.min_similarity, chunks=chunks
        )


def format_retrieval(result: RetrievalResult) -> str:
    """Human-readable table of retrieval results for the terminal."""
    lines = [
        f"Question: {result.question}",
        f"Top-k: {result.top_k} | metric: cosine | similarity = 1 - distance (higher = better) "
        f"| relevance threshold: similarity >= {result.min_similarity}",
    ]
    if not result.chunks:
        lines.append("  (no results - the vector store is empty)")
        return "\n".join(lines)
    lines.append(
        f"  {'#':<3}{'similarity':>10} {'distance':>9}  {'chunk_id':<13} {'policy':<7} title / section"
    )
    for c in result.chunks:
        flag = "" if c.similarity >= result.min_similarity else "  [below threshold]"
        lines.append(
            f"  {c.rank:<3}{c.similarity:>10.4f} {c.distance:>9.4f}  {c.chunk_id:<13} {c.policy_id:<7} "
            f"{c.title} / {c.section}{flag}"
        )
    if not result.has_relevant_context:
        lines.append("  -> No chunk meets the relevance threshold: the question will be refused.")
    return "\n".join(lines)
