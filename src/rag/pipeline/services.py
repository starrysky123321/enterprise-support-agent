from __future__ import annotations

from dataclasses import dataclass, replace
from uuid import uuid4

from src.rag.embeddings.interface import EmbeddingProvider
from src.rag.ingestion.chunker import ChunkingStrategyRegistry
from src.rag.ingestion.chunking.fixed_window import FixedWindowChunkingStrategy
from src.rag.ingestion.pdf_extractor import PDFExtractor
from src.rag.models import RetrievedChunk
from src.rag.models import RetrievalFilter
from src.rag.keyword import KeywordIndex
from src.rag.retrieval import reciprocal_rank_fusion
from src.rag.reranker import Reranker
from src.rag.vectorstore.interface import VectorStore
from src.shared.tracing import TraceContext, chunk_metadata, trace_event


@dataclass(slots=True)
class IngestionResult:
    doc_id: str
    chunks_ingested: int
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int


@dataclass(slots=True)
class PDFIngestionResult:
    doc_id: str
    chunks_ingested: int
    pages_total: int
    pages_ingested: int
    skipped_pages: list[int]
    warnings: list[str]
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int


class RAGIngestionService:
    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        chunking_registry: ChunkingStrategyRegistry | None = None,
        default_chunking_strategy: str = "fixed_window",
        chunk_size: int,
        chunk_overlap: int,
        pdf_extractor: PDFExtractor | None = None,
        pdf_max_pages: int = 300,
        keyword_index: KeywordIndex | None = None,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._chunking_registry = chunking_registry or ChunkingStrategyRegistry(
            [FixedWindowChunkingStrategy()]
        )
        self._default_chunking_strategy = default_chunking_strategy
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._pdf_extractor = pdf_extractor
        self._pdf_max_pages = pdf_max_pages
        self._keyword_index = keyword_index

    async def ingest_text(
        self,
        *,
        text: str,
        source: str | None = None,
        doc_id: str | None = None,
        chunking_strategy: str | None = None,
        metadata: dict[str, str | None] | None = None,
    ) -> IngestionResult:
        resolved_doc_id = doc_id or str(uuid4())
        resolved_source = source or "inline-text"
        resolved_strategy_name = chunking_strategy or self._default_chunking_strategy
        strategy = self._chunking_registry.resolve(resolved_strategy_name)

        chunks = strategy.chunk(
            text=text,
            doc_id=resolved_doc_id,
            source=resolved_source,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )
        self._apply_metadata(chunks, metadata)
        embeddings = await self._embedding_provider.embed_documents(
            [chunk.text for chunk in chunks]
        )
        await self._vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)
        if self._keyword_index is not None:
            await self._keyword_index.upsert_chunks(chunks=chunks)

        return IngestionResult(
            doc_id=resolved_doc_id,
            chunks_ingested=len(chunks),
            chunking_strategy=strategy.name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    async def ingest_pdf(
        self,
        *,
        pdf_bytes: bytes,
        source: str | None = None,
        doc_id: str | None = None,
        chunking_strategy: str | None = None,
        metadata: dict[str, str | None] | None = None,
    ) -> PDFIngestionResult:
        if self._pdf_extractor is None:
            raise RuntimeError("PDF extractor is not configured.")

        resolved_doc_id = doc_id or str(uuid4())
        resolved_source = source or "uploaded-pdf"
        resolved_strategy_name = chunking_strategy or self._default_chunking_strategy
        strategy = self._chunking_registry.resolve(resolved_strategy_name)
        extraction = await self._pdf_extractor.extract(
            pdf_bytes=pdf_bytes,
            max_pages=self._pdf_max_pages,
        )

        if extraction.pages_ingested == 0:
            raise ValueError("No extractable pages were found in this PDF.")

        chunks = []
        global_index = 0
        for segment in extraction.segments:
            segment_chunks = strategy.chunk(
                text=segment.text,
                doc_id=resolved_doc_id,
                source=resolved_source,
                chunk_size=self._chunk_size,
                chunk_overlap=self._chunk_overlap,
                page_number=segment.page_number,
            )
            for chunk in segment_chunks:
                chunk.chunk_id = f"chunk-{global_index}"
                global_index += 1
                chunks.append(chunk)

        if not chunks:
            raise ValueError("No chunks were generated from extracted PDF content.")

        self._apply_metadata(chunks, metadata)

        embeddings = await self._embedding_provider.embed_documents(
            [chunk.text for chunk in chunks]
        )
        await self._vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)
        if self._keyword_index is not None:
            await self._keyword_index.upsert_chunks(chunks=chunks)

        return PDFIngestionResult(
            doc_id=resolved_doc_id,
            chunks_ingested=len(chunks),
            pages_total=extraction.pages_total,
            pages_ingested=extraction.pages_ingested,
            skipped_pages=extraction.skipped_pages,
            warnings=extraction.warnings,
            chunking_strategy=strategy.name,
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap,
        )

    @staticmethod
    def _apply_metadata(chunks: list, metadata: dict[str, str | None] | None) -> None:
        if not metadata:
            return
        allowed = {
            "document_type", "product_name", "product_version", "department",
            "knowledge_space_id", "visibility",
        }
        for chunk in chunks:
            for key, value in metadata.items():
                if key in allowed and value is not None:
                    setattr(chunk, key, value)


class RAGRetrievalService:
    RERANK_MAX_ATTEMPTS = 2
    RERANK_EXHAUSTED_ERROR = "Reranker failed after one retry."

    def __init__(
        self,
        *,
        embedding_provider: EmbeddingProvider,
        vector_store: VectorStore,
        default_top_k: int,
        prefetch_k: int,
        reranker: Reranker | None = None,
        keyword_index: KeywordIndex | None = None,
        dense_enabled: bool = True,
        bm25_enabled: bool = True,
        dense_fetch_k: int | None = None,
        bm25_fetch_k: int = 40,
        rrf_k: int = 60,
        candidate_k: int = 40,
        reranker_fail_open: bool = False,
    ) -> None:
        self._embedding_provider = embedding_provider
        self._vector_store = vector_store
        self._default_top_k = default_top_k
        self._prefetch_k = prefetch_k
        self._reranker = reranker
        self._keyword_index = keyword_index
        self._dense_enabled = dense_enabled
        self._bm25_enabled = bm25_enabled
        self._dense_fetch_k = dense_fetch_k or prefetch_k
        self._bm25_fetch_k = bm25_fetch_k
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k
        self._reranker_fail_open = reranker_fail_open

    async def retrieve(
        self,
        *,
        query: str,
        top_k: int | None = None,
        doc_id: str | None = None,
        trace_context: TraceContext | None = None,
        filters: RetrievalFilter | None = None,
    ) -> list[RetrievedChunk]:
        final_k = top_k or self._default_top_k
        if final_k < 1:
            raise ValueError("top_k must be >= 1")
        if filters is not None and filters.deny_all:
            return []

        effective_filters = filters
        if doc_id:
            if filters is None:
                effective_filters = RetrievalFilter(allowed_doc_ids=(doc_id,))
            else:
                allowed = (
                    tuple(item for item in filters.allowed_doc_ids if item == doc_id)
                    if filters.allowed_doc_ids
                    else (doc_id,)
                )
                effective_filters = replace(filters, allowed_doc_ids=allowed)
                if not allowed:
                    return []

        candidate_k = max(self._candidate_k, final_k)
        trace_event(
            "retrieval.started",
            trace_context=trace_context,
            query=query,
            top_k=final_k,
            prefetch_k=candidate_k,
            reranker_enabled=self._reranker is not None,
            reranker_model=(
                getattr(self._reranker, "model_name", None)
                if self._reranker is not None
                else None
            ),
        )
        ranked_lists: dict[str, list[RetrievedChunk]] = {}
        if self._dense_enabled:
            query_embedding = await self._embedding_provider.embed_query(query)
            dense_k = max(self._dense_fetch_k, final_k)
            try:
                dense = await self._vector_store.similarity_search(
                    query_embedding=query_embedding, top_k=dense_k,
                    doc_id=doc_id, filters=effective_filters,
                )
            except TypeError:
                # Compatibility for third-party adapters compiled against the old interface.
                dense = await self._vector_store.similarity_search(
                    query_embedding=query_embedding, top_k=dense_k, doc_id=doc_id,
                )
                if effective_filters:
                    dense = [chunk for chunk in dense if effective_filters.matches(chunk)]
            ranked_lists["dense"] = dense
        if self._bm25_enabled and self._keyword_index is not None:
            ranked_lists["bm25"] = await self._keyword_index.search(
                query=query, top_k=max(self._bm25_fetch_k, final_k), filters=effective_filters,
            )
        candidates = reciprocal_rank_fusion(
            ranked_lists, rrf_k=self._rrf_k, limit=candidate_k
        ) if len(ranked_lists) > 1 else next(iter(ranked_lists.values()), [])[:candidate_k]
        trace_event(
            "retrieval.vector_search.completed",
            trace_context=trace_context,
            query=query,
            top_k=final_k,
            prefetch_k=candidate_k,
            candidate_count=len(candidates),
            candidates=chunk_metadata(candidates),
        )
        if not candidates:
            trace_event(
                "retrieval.completed",
                trace_context=trace_context,
                query=query,
                returned_count=0,
                reranker_status="disabled" if self._reranker is None else None,
                results=[],
            )
            return []
        if self._reranker is None:
            results = candidates[:final_k]
            trace_event(
                "retrieval.completed",
                trace_context=trace_context,
                query=query,
                returned_count=len(results),
                reranker_status="disabled",
                results=chunk_metadata(results),
            )
            return results

        for attempt in range(1, self.RERANK_MAX_ATTEMPTS + 1):
            try:
                reranked = await self._reranker.rerank(
                    query=query,
                    chunks=candidates,
                    top_n=final_k,
                )
                results = reranked[:final_k]
                reranker_status = (
                    "succeeded_first_try" if attempt == 1 else "succeeded_on_retry"
                )
                trace_event(
                    "retrieval.rerank.succeeded",
                    trace_context=trace_context,
                    query=query,
                    reranker_status=reranker_status,
                    reranker_attempt_count=attempt,
                    returned_count=len(results),
                    results=chunk_metadata(results),
                )
                trace_event(
                    "retrieval.completed",
                    trace_context=trace_context,
                    query=query,
                    returned_count=len(results),
                    reranker_status=reranker_status,
                    results=chunk_metadata(results),
                )
                return results
            except Exception as exc:
                if attempt >= self.RERANK_MAX_ATTEMPTS:
                    trace_event(
                        "retrieval.rerank.failed",
                        trace_context=trace_context,
                        query=query,
                        reranker_status="failed",
                        reranker_attempt_count=attempt,
                        error=str(exc),
                    )
                    if self._reranker_fail_open:
                        trace_event(
                            "retrieval.rerank.degraded", trace_context=trace_context,
                            error=str(exc), returned_count=min(len(candidates), final_k),
                        )
                        return candidates[:final_k]
                    raise RuntimeError(self.RERANK_EXHAUSTED_ERROR) from exc
                trace_event(
                    "retrieval.rerank.retry",
                    trace_context=trace_context,
                    query=query,
                    reranker_attempt_count=attempt,
                    error=str(exc),
                )

        raise RuntimeError(self.RERANK_EXHAUSTED_ERROR)
