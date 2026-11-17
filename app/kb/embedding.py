"""Embedding client: thin wrapper around ``OllamaEmbeddings`` pointed at BGE-M3."""

from __future__ import annotations

from langchain_ollama import OllamaEmbeddings

from app.core.config import get_settings


class Embedder:
    """Turn texts into fixed-width embeddings via an ``embed_documents`` backend.

    The default backend is ``OllamaEmbeddings`` configured from ``Settings``
    (``embedding_model`` / ``embedding_base_url``), i.e. Ollama's BGE-M3 behind
    its native ``/api/embed`` endpoint. ``embed_texts`` returns one vector per
    input text and guards against dimension drift so downstream vector-store
    writes never silently store the wrong width.
    """

    def __init__(self, underlying=None, dim: int = 1024):
        if underlying is None:
            settings = get_settings()
            underlying = OllamaEmbeddings(
                model=settings.embedding_model,
                base_url=settings.embedding_base_url,
            )
        self._underlying = underlying
        self.dim = dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors = self._underlying.embed_documents(texts)
        for vector in vectors:
            if len(vector) != self.dim:
                raise ValueError(
                    f"embedding dimension mismatch: expected {self.dim}, got {len(vector)}"
                )
        return vectors
