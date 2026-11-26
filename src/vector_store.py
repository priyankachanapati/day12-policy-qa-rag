"""ChromaDB vector store for policy chunks.

The collection uses **cosine distance** (``distance = 1 - cosine_similarity``),
so distance 0 means identical direction and lower distance = better match.
Retrieval converts this to ``similarity = 1 - distance`` (higher = better).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import chromadb

from .ingestion import Chunk

DISTANCE_METRIC = "cosine"


@dataclass(frozen=True)
class StoredMatch:
    """A raw match returned by the vector store."""

    chunk_id: str
    text: str
    metadata: dict
    distance: float


class PolicyVectorStore:
    """Persistent ChromaDB collection holding chunk text, embeddings and metadata."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        self.persist_dir = Path(persist_dir)
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(path=str(self.persist_dir))
        self._collection = self._get_or_create()

    def _get_or_create(self):
        # embedding_function=None: embeddings are always supplied explicitly by
        # our embedder, so the model used is visible and consistent.
        return self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": DISTANCE_METRIC},
            embedding_function=None,
        )

    def reset(self) -> None:
        """Delete and recreate the collection so re-ingestion never duplicates data."""
        existing = [c.name for c in self._client.list_collections()]
        if self.collection_name in existing:
            self._client.delete_collection(self.collection_name)
        self._collection = self._get_or_create()

    def add_chunks(self, chunks: list[Chunk], embeddings: list[list[float]]) -> None:
        """Store chunks with their embeddings and metadata."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[dict(c.metadata) for c in chunks],
        )

    def count(self) -> int:
        """Number of chunks stored."""
        return self._collection.count()

    def query(self, query_embedding: list[float], top_k: int) -> list[StoredMatch]:
        """Return up to ``top_k`` nearest chunks, best (lowest distance) first."""
        if self.count() == 0:
            return []
        result = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            StoredMatch(chunk_id=cid, text=doc, metadata=meta, distance=float(dist))
            for cid, doc, meta, dist in zip(
                result["ids"][0],
                result["documents"][0],
                result["metadatas"][0],
                result["distances"][0],
                strict=True,
            )
        ]
