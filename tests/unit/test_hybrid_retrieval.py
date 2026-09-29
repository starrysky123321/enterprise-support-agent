import asyncio

from src.infrastructure.keyword import SQLiteBM25Index
from src.rag.models import RAGChunk, RetrievedChunk, RetrievalFilter
from src.rag.retrieval import reciprocal_rank_fusion
from src.rag.pipeline import RAGRetrievalService
from src.infrastructure.llm.local_embeddings import LocalHashEmbeddingProvider


class _UnusedVectorStore:
    async def similarity_search(self, **kwargs):
        raise AssertionError("dense path should be disabled")


class _UnusedEmbedding:
    async def embed_query(self, text):
        raise AssertionError("deny_all must stop before embedding")


def _retrieved(doc_id: str, chunk_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(doc_id=doc_id, chunk_id=chunk_id, source="s", text="t", score=score)


def test_rrf_merges_duplicates_and_tracks_routes():
    fused = reciprocal_rank_fusion(
        {
            "dense": [_retrieved("d", "a", 0.9), _retrieved("d", "b", 0.8)],
            "bm25": [_retrieved("d", "b", 8.0), _retrieved("d", "c", 7.0)],
        },
        rrf_k=60,
    )
    assert [item.chunk_id for item in fused] == ["b", "a", "c"]
    assert fused[0].retrieval_sources == ("bm25", "dense")
    assert fused[0].raw_scores == {"dense": 0.8, "bm25": 8.0}


def test_sqlite_bm25_applies_workspace_filter_before_return(tmp_path):
    index = SQLiteBM25Index(database_path=str(tmp_path / "bm25.db"))
    chunks = [
        RAGChunk(
            doc_id="allowed", chunk_id="c1", source="ticket", text="E_CONN_TIMEOUT failure",
            knowledge_space_id="space-a", document_type="ticket",
        ),
        RAGChunk(
            doc_id="forbidden", chunk_id="c1", source="ticket", text="E_CONN_TIMEOUT secret",
            knowledge_space_id="space-b", document_type="ticket",
        ),
    ]

    async def run():
        await index.upsert_chunks(chunks=chunks)
        await index.upsert_chunks(chunks=chunks)  # retry is idempotent
        return await index.search(
            query="E_CONN_TIMEOUT", top_k=10,
            filters=RetrievalFilter(knowledge_space_id="space-a", allowed_doc_ids=("allowed",)),
        )

    results = asyncio.run(run())
    assert [(item.doc_id, item.chunk_id) for item in results] == [("allowed", "c1")]
    assert results[0].retrieval_sources == ("bm25",)


def test_doc_id_scope_is_applied_to_bm25_before_fusion(tmp_path):
    index = SQLiteBM25Index(database_path=str(tmp_path / "bm25.db"))
    chunks = [
        RAGChunk(doc_id="allowed", chunk_id="c1", source="a", text="CFG-3207 setting"),
        RAGChunk(doc_id="other", chunk_id="c1", source="b", text="CFG-3207 secret setting"),
    ]
    service = RAGRetrievalService(
        embedding_provider=LocalHashEmbeddingProvider(), vector_store=_UnusedVectorStore(),
        default_top_k=5, prefetch_k=5, keyword_index=index,
        dense_enabled=False, bm25_enabled=True,
    )

    async def run():
        await index.upsert_chunks(chunks=chunks)
        return await service.retrieve(query="CFG-3207", doc_id="allowed")

    results = asyncio.run(run())
    assert [item.doc_id for item in results] == ["allowed"]


def test_explicit_deny_all_stops_before_any_retrieval_backend():
    service = RAGRetrievalService(
        embedding_provider=_UnusedEmbedding(), vector_store=_UnusedVectorStore(),
        default_top_k=5, prefetch_k=5, dense_enabled=True, bm25_enabled=False,
    )

    results = asyncio.run(service.retrieve(
        query="secret", filters=RetrievalFilter(deny_all=True),
    ))

    assert results == []
