import asyncio

import pytest

from src.infrastructure.reranker.local_cross_encoder import LocalCrossEncoderReranker
from src.rag.models import RetrievedChunk


def _chunks() -> list[RetrievedChunk]:
    return [
        RetrievedChunk(
            doc_id="doc-a", chunk_id="chunk-0", source="a.md", text="unrelated", score=0.9,
        ),
        RetrievedChunk(
            doc_id="doc-b", chunk_id="chunk-0", source="b.md", text="timeout E100", score=0.1,
            retrieval_sources=("bm25", "dense"),
        ),
    ]


def test_local_cross_encoder_ranks_and_preserves_provenance(monkeypatch):
    class FakeModel:
        def __init__(self, model_name: str):
            self.model_name = model_name

        def predict(self, pairs):
            assert pairs[1] == ("E100 timeout", "timeout E100")
            return [0.1, 0.95]

    monkeypatch.setattr(
        "src.infrastructure.reranker.local_cross_encoder._CrossEncoder", FakeModel,
    )
    reranker = LocalCrossEncoderReranker(model="test-cross-encoder")

    result = asyncio.run(reranker.rerank(query="E100 timeout", chunks=_chunks(), top_n=1))

    assert result[0].doc_id == "doc-b"
    assert result[0].score == pytest.approx(0.95)
    assert result[0].retrieval_sources == ("bm25", "dense")


def test_local_cross_encoder_rejects_invalid_top_n(monkeypatch):
    monkeypatch.setattr(
        "src.infrastructure.reranker.local_cross_encoder._CrossEncoder", object,
    )
    reranker = LocalCrossEncoderReranker(model="test-cross-encoder")
    with pytest.raises(ValueError, match="top_n"):
        asyncio.run(reranker.rerank(query="q", chunks=_chunks(), top_n=0))
