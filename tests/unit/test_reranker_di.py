from types import SimpleNamespace

import pytest

from src.api.v1 import dependencies as deps
from src.settings.config import settings


@pytest.fixture(autouse=True)
def _restore_settings_and_cache():
    fields = [
        "reranker_enabled",
        "reranker_provider",
        "reranker_model",
        "reranker_api_key",
        "reranker_timeout_s",
        "embedding_provider",
        "embedding_model",
        "openai_key",
        "rag_top_k",
        "rag_prefetch_k",
    ]
    snapshot = {field: getattr(settings, field) for field in fields}
    deps.get_reranker.cache_clear()
    deps.get_rag_retrieval_service.cache_clear()
    try:
        yield
    finally:
        for field, value in snapshot.items():
            setattr(settings, field, value)
        deps.get_reranker.cache_clear()
        deps.get_rag_retrieval_service.cache_clear()


def test_get_reranker_returns_none_when_disabled():
    settings.reranker_enabled = False

    reranker = deps.get_reranker()

    assert reranker is None


def test_get_reranker_missing_key_raises():
    settings.reranker_enabled = True
    settings.reranker_model = "rerank-v4.0-fast"
    settings.reranker_api_key = None

    with pytest.raises(RuntimeError, match="RERANKER_API_KEY"):
        deps.get_reranker()


def test_get_reranker_returns_cohere_reranker(monkeypatch):
    class FakeCohereReranker:
        def __init__(self, *, api_key: str, model: str, timeout_s: float) -> None:
            self.api_key = api_key
            self.model = model
            self.timeout_s = timeout_s

    settings.reranker_enabled = True
    settings.reranker_provider = "cohere"
    settings.reranker_model = "rerank-v4.0-fast"
    settings.reranker_api_key = "cohere-key"

    monkeypatch.setitem(
        __import__("sys").modules,
        "src.infrastructure.reranker",
        SimpleNamespace(CohereReranker=FakeCohereReranker),
    )

    reranker = deps.get_reranker()

    assert isinstance(reranker, FakeCohereReranker)
    assert reranker.model == "rerank-v4.0-fast"
    assert reranker.api_key == "cohere-key"
    assert reranker.timeout_s == settings.reranker_timeout_s


def test_get_reranker_returns_local_cross_encoder(monkeypatch):
    class FakeLocalReranker:
        def __init__(self, *, model: str, timeout_s: float) -> None:
            self.model = model
            self.timeout_s = timeout_s

    settings.reranker_enabled = True
    settings.reranker_provider = "local"
    settings.reranker_model = "cross-encoder/test"
    settings.reranker_api_key = None
    monkeypatch.setitem(
        __import__("sys").modules,
        "src.infrastructure.reranker",
        SimpleNamespace(LocalCrossEncoderReranker=FakeLocalReranker),
    )

    reranker = deps.get_reranker()

    assert isinstance(reranker, FakeLocalReranker)
    assert reranker.model == "cross-encoder/test"
