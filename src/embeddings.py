"""Embedding generation.

Uses the ``all-MiniLM-L6-v2`` sentence-transformer model, run locally as an ONNX
model through ChromaDB's built-in embedding function. It needs no API key and
produces 384-dimensional vectors. The model files (~80 MB) are downloaded to
``~/.cache/chroma`` on first use.
"""

from __future__ import annotations

from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2 (ONNX, run locally via ChromaDB)"


class EmbeddingError(RuntimeError):
    """Raised when embeddings cannot be generated."""


class LocalEmbedder:
    """Thin wrapper around the local MiniLM embedding model."""

    model_name = EMBEDDING_MODEL_NAME

    def __init__(self) -> None:
        self._model = ONNXMiniLM_L6_V2()

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns one vector per input text."""
        if not texts:
            return []
        if any(not t or not t.strip() for t in texts):
            raise EmbeddingError("Cannot embed empty text")
        try:
            vectors = self._model(texts)
        except Exception as exc:  # model download / runtime failures
            raise EmbeddingError(f"Embedding generation failed: {exc}") from exc
        return [[float(x) for x in v] for v in vectors]

    def embed_one(self, text: str) -> list[float]:
        """Embed a single text."""
        return self.embed([text])[0]
