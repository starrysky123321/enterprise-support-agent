from __future__ import annotations

import asyncio
from typing import Any

from src.rag.models import RetrievedChunk
from src.rag.reranker import Reranker

_CrossEncoder: Any
try:
    from sentence_transformers import CrossEncoder
    _CrossEncoder = CrossEncoder
except ImportError:  # pragma: no cover - exercised only in minimal deployments
    _CrossEncoder = None


class LocalCrossEncoderReranker(Reranker):
    """Lazy local Cross-Encoder adapter suitable for offline deployments."""

    def __init__(self, *, model: str, timeout_s: float = 20.0) -> None:
        if _CrossEncoder is None:
            raise RuntimeError(
                "sentence-transformers is not installed; local reranking is unavailable."
            )
        self._model_name = model
        self._timeout_s = timeout_s
        self._model: Any | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def _get_model(self) -> Any:
        if self._model is None:
            self._model = _CrossEncoder(self._model_name)
        return self._model

    async def rerank(
        self, *, query: str, chunks: list[RetrievedChunk], top_n: int,
    ) -> list[RetrievedChunk]:
        if top_n < 1:
            raise ValueError("top_n must be >= 1")
        if not chunks:
            return []
        return await asyncio.wait_for(
            asyncio.to_thread(self._rerank_sync, query=query, chunks=chunks, top_n=top_n),
            timeout=self._timeout_s,
        )

    def _rerank_sync(
        self, *, query: str, chunks: list[RetrievedChunk], top_n: int,
    ) -> list[RetrievedChunk]:
        scores = self._get_model().predict([(query, chunk.text) for chunk in chunks])
        if len(scores) != len(chunks):
            raise ValueError("Cross-Encoder returned a score count that does not match candidates.")
        ranked = sorted(
            zip(chunks, scores, strict=True), key=lambda item: float(item[1]), reverse=True,
        )
        return [
            RetrievedChunk(
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                text=chunk.text,
                score=float(score),
                page_number=chunk.page_number,
                document_type=chunk.document_type,
                product_name=chunk.product_name,
                product_version=chunk.product_version,
                department=chunk.department,
                knowledge_space_id=chunk.knowledge_space_id,
                visibility=chunk.visibility,
                retrieval_sources=chunk.retrieval_sources,
                raw_scores=chunk.raw_scores,
            )
            for chunk, score in ranked[: min(top_n, len(ranked))]
        ]
