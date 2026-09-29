from __future__ import annotations

from abc import ABC, abstractmethod

from src.rag.models import RAGChunk, RetrievedChunk, RetrievalFilter


class KeywordIndex(ABC):
    @abstractmethod
    async def upsert_chunks(self, *, chunks: list[RAGChunk]) -> None:
        """Idempotently index chunks for keyword retrieval."""

    @abstractmethod
    async def search(
        self, *, query: str, top_k: int, filters: RetrievalFilter | None = None
    ) -> list[RetrievedChunk]:
        """Return BM25-ranked chunks after applying filters inside the query."""

    @abstractmethod
    async def delete_by_doc_id(self, *, doc_id: str) -> None:
        """Remove every keyword entry belonging to a document."""
