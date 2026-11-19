import asyncio
from dataclasses import replace

import pytest

from src.infrastructure.keyword import SQLiteBM25Index
from src.rag.models import RAGChunk, RetrievedChunk, RetrievalFilter
from src.rag.retrieval import reciprocal_rank_fusion
from src.rag.retrieval.error_codes import find_error_code_definition, lookup_error_code
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


@pytest.mark.parametrize("with_reranker", [False, True])
def test_error_code_definition_survives_candidate_cutoff_and_reranking(with_reranker):
    definition = replace(
        _retrieved("release-32", "c2", 0.9),
        text="CFG-3207 indicates the legacy keepalive.enabled key is still present.",
    )
    distractors = [
        replace(
            _retrieved(f"ticket-{index:02}", "c1", 0.8),
            text="The team saw CFG-3207 during a rollout and opened a ticket.",
        )
        for index in range(12)
    ]

    class KeywordHits:
        async def search(self, **kwargs):
            assert kwargs["query"] == "CFG-3207"
            return [definition, *distractors]

    class DenseHits:
        async def similarity_search(self, **kwargs):
            return distractors

    class DropsDefinition:
        async def rerank(self, *, query, chunks, top_n):
            assert chunks[0].doc_id == "release-32"
            return chunks[1:top_n + 1]

    baseline = reciprocal_rank_fusion(
        {"bm25": [definition, *distractors], "dense": distractors}, limit=5,
    )
    assert definition.doc_id not in {chunk.doc_id for chunk in baseline}
    service = RAGRetrievalService(
        embedding_provider=LocalHashEmbeddingProvider(), vector_store=DenseHits(),
        default_top_k=3, prefetch_k=20, candidate_k=5, keyword_index=KeywordHits(),
        reranker=DropsDefinition() if with_reranker else None,
    )

    result = asyncio.run(service.retrieve(query="CFG-3207是什么意思？"))

    assert [chunk.doc_id for chunk in result] == [
        "release-32", distractors[0].doc_id, distractors[1].doc_id,
    ]
    assert result[0].retrieval_sources == ("bm25",)
    assert result[0].raw_scores == {"bm25": 0.9}


@pytest.mark.parametrize("query", ["cfg-3207", "What does `CFG-3207` mean?", "错误码：CFG-3207 是什么意思？"])
def test_definition_is_exact_and_stays_in_scope(query):
    allowed = replace(
        _retrieved("public-manual", "c1", 0.7),
        text="错误码 CFG-3207 表示旧的保活配置仍然存在。", product_version="3.2",
    )
    candidates = [
        replace(allowed, doc_id="private", text="CFG-3207 means a private diagnostic."),
        replace(allowed, doc_id="old", product_version="3.1"),
        replace(allowed, text="CFG-32070 means a different issue."),
        replace(allowed, text="The operator saw CFG-3207 during an upgrade."),
        replace(allowed, text="CFG-3207 was returned. Contact the operator."),
        allowed,
    ]
    filters = RetrievalFilter(allowed_doc_ids=("public-manual", "old"), product_version="3.2")

    assert find_error_code_definition(lookup_error_code(query), candidates, filters) is allowed


@pytest.mark.parametrize("query", [
    "How should I diagnose CFG-3207 after an upgrade?",
    "CFG-3207 和 TLS-3210 是什么意思？",
])
def test_no_reservation_for_broader_queries(query):
    definition = replace(_retrieved("r", "c1", 0.9), text="CFG-3207 means legacy keys are used.")
    assert find_error_code_definition(lookup_error_code(query), [definition]) is None
