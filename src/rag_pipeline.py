"""End-to-end RAG pipeline and command-line interface.

Usage (from the project root, with the virtual environment active):

    python -m src.rag_pipeline ingest
    python -m src.rag_pipeline retrieve "How many days of annual leave do I get?" [--top-k 4]
    python -m src.rag_pipeline ask "How many days of annual leave do I get?"
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from dataclasses import dataclass

from .citations import CitationReport, validate_citations
from .config import ConfigurationError, Settings, load_settings
from .embeddings import EmbeddingError, LocalEmbedder
from .generation import (
    REFUSAL_MESSAGE,
    SYSTEM_PROMPT,
    GenerationError,
    GroqGenerator,
    LLMClient,
    build_user_prompt,
    is_refusal,
)
from .ingestion import chunk_documents, load_documents
from .retrieval import RetrievalResult, Retriever, format_retrieval
from .vector_store import DISTANCE_METRIC, PolicyVectorStore

# Answer statuses
ANSWERED = "answered"  # grounded answer with only valid citations
REFUSED = "refused"  # LLM judged the retrieved context insufficient
REFUSED_NO_CONTEXT = "refused_no_context"  # nothing relevant retrieved; LLM not called
REJECTED = "rejected_invalid_citations"  # answer withheld: missing/invalid/malformed citations


@dataclass
class IngestionSummary:
    document_count: int
    chunk_count: int
    chunks_per_document: dict[str, int]
    chunk_size: int
    chunk_overlap: int
    embedding_model: str
    embedding_dimensions: int
    vector_db: str
    stored_count: int

    def format(self) -> str:
        per_doc = "\n".join(f"    {pid}: {n} chunks" for pid, n in self.chunks_per_document.items())
        return (
            "=== Ingestion summary ===\n"
            f"  Source documents : {self.document_count}\n"
            f"  Chunks generated : {self.chunk_count}\n{per_doc}\n"
            f"  Chunking config  : chunk_size={self.chunk_size} chars, overlap={self.chunk_overlap} chars "
            "(paragraph-aware)\n"
            f"  Embedding model  : {self.embedding_model} ({self.embedding_dimensions} dimensions)\n"
            f"  Vector database  : {self.vector_db}\n"
            f"  Chunks stored    : {self.stored_count}"
        )


@dataclass
class RAGAnswer:
    question: str
    status: str
    answer: str  # what is shown to the user
    raw_llm_answer: str | None
    retrieval: RetrievalResult
    citations: CitationReport | None
    llm_model: str | None

    @property
    def is_refusal(self) -> bool:
        return self.status in (REFUSED, REFUSED_NO_CONTEXT)


class RAGPipeline:
    """Ingest documents, retrieve context and generate cited answers."""

    def __init__(self, settings: Settings, llm: LLMClient | None = None) -> None:
        self.settings = settings
        self.embedder = LocalEmbedder()
        self.store = PolicyVectorStore(settings.chroma_dir, settings.collection_name)
        self.retriever = Retriever(self.embedder, self.store, settings.top_k, settings.min_similarity)
        self._llm = llm

    @property
    def llm(self) -> LLMClient:
        """Lazily create the real Groq client (only needed for answer generation)."""
        if self._llm is None:
            self._llm = GroqGenerator(
                api_key=self.settings.require_groq_key(),
                model=self.settings.llm_model,
                temperature=self.settings.llm_temperature,
            )
        return self._llm

    def ingest(self) -> IngestionSummary:
        """Load, chunk, embed and store all policy documents (replacing previous data)."""
        docs = load_documents(self.settings.policy_dir)
        chunks = chunk_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        embeddings = self.embedder.embed([c.embedding_text for c in chunks])
        self.store.reset()
        self.store.add_chunks(chunks, embeddings)
        return IngestionSummary(
            document_count=len(docs),
            chunk_count=len(chunks),
            chunks_per_document=dict(Counter(c.metadata["policy_id"] for c in chunks)),
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            embedding_model=self.embedder.model_name,
            embedding_dimensions=len(embeddings[0]) if embeddings else 0,
            vector_db=f"ChromaDB (persistent, collection '{self.settings.collection_name}', "
            f"{DISTANCE_METRIC} distance)",
            stored_count=self.store.count(),
        )

    def ask(self, question: str) -> RAGAnswer:
        """Answer ``question`` with citations, or refuse if the policies do not support an answer."""
        retrieval = self.retriever.retrieve(question)
        context = retrieval.relevant_chunks
        if not context:
            return RAGAnswer(question, REFUSED_NO_CONTEXT, REFUSAL_MESSAGE, None, retrieval, None, None)

        raw = self.llm.complete(SYSTEM_PROMPT, build_user_prompt(question, context))
        report = validate_citations(raw, context)

        if is_refusal(raw):
            return RAGAnswer(question, REFUSED, REFUSAL_MESSAGE, raw, retrieval, report, self.llm.model)
        if not report.all_valid:
            withheld = (
                "Answer withheld: the generated answer failed citation validation "
                "(missing, invalid or malformed citations), so it cannot be shown as a grounded answer."
            )
            return RAGAnswer(question, REJECTED, withheld, raw, retrieval, report, self.llm.model)
        return RAGAnswer(question, ANSWERED, raw, raw, retrieval, report, self.llm.model)


def format_answer(result: RAGAnswer) -> str:
    """Human-readable answer report for the terminal."""
    lines = [format_retrieval(result.retrieval), "", f"LLM: {result.llm_model or '(not called)'}"]
    lines.append(f"Status: {result.status.upper()}")
    lines.append(f"Answer: {result.answer}")
    if result.status == REJECTED and result.raw_llm_answer:
        lines.append(f"Raw LLM output (not trusted): {result.raw_llm_answer}")
    r = result.citations
    if r is not None:
        lines.append("Citation validation:")
        lines.append(f"  valid     : {', '.join(map(str, r.valid)) or 'none'}")
        lines.append("  invalid   : " + (", ".join(f"{c} ({why})" for c, why in r.invalid) or "none"))
        lines.append(f"  malformed : {', '.join(r.malformed) or 'none'}")
        if result.status == ANSWERED:
            lines.append(f"  uncited sentences (informational): {len(r.uncited_sentences)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Policy Q&A RAG v1")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("ingest", help="Load, chunk, embed and store the policy documents")
    p_ret = sub.add_parser("retrieve", help="Show top-k retrieved chunks with scores")
    p_ret.add_argument("question")
    p_ret.add_argument("--top-k", type=int, default=None)
    p_ask = sub.add_parser("ask", help="Answer a question with citations (uses the Groq LLM)")
    p_ask.add_argument("question")
    args = parser.parse_args(argv)

    try:
        pipeline = RAGPipeline(load_settings())
        if args.command == "ingest":
            print(pipeline.ingest().format())
            return 0
        if pipeline.store.count() == 0:
            print("The vector store is empty. Run: python -m src.rag_pipeline ingest", file=sys.stderr)
            return 1
        if args.command == "retrieve":
            print(format_retrieval(pipeline.retriever.retrieve(args.question, args.top_k)))
        else:
            print(format_answer(pipeline.ask(args.question)))
        return 0
    except (ConfigurationError, GenerationError, EmbeddingError, FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
