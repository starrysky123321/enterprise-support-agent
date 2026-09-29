from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from src.agents.ask_pipeline import AgentAskPipeline
from src.agents.service import AgentCitation, AgentResult
from src.shared.tracing import TRACE_LOGGER_NAME
from src.settings.config import settings


@dataclass
class _Doc:
    id: str
    owner_user_id: object
    last_indexed_at: datetime | None


class _DocsRepo:
    async def get_owned_document(self, *, owner_user_id, doc_id: str, include_deleted: bool = False):
        return _Doc(id=doc_id, owner_user_id=owner_user_id, last_indexed_at=datetime.now(timezone.utc))


class _Agent:
    async def run(
        self,
        *,
        question: str,
        doc_id: str,
        session_id: str | None = None,
        user_id: str | None = None,
        request_id: str | None = None,
    ):
        return AgentResult(
            answer="answer",
            steps=1,
            tools_used=["retrieve_context"],
            status="ok",
            citations=[
                AgentCitation(
                    source="inline",
                    doc_id=doc_id,
                    chunk_id="chunk-1",
                    snippet="snippet",
                    page_number=None,
                )
            ],
        )


class _LLM:
    @property
    def model_name(self) -> str:
        return "model-x"


class _Refiner:
    class _Out:
        refined_query = "refined q"

    async def refine(self, *, question: str, doc_id: str):
        return self._Out()


class _Embed:
    def __init__(self) -> None:
        self.calls = 0

    async def embed_query(self, text: str):
        self.calls += 1
        return [0.1, 0.2]


class _Cache:
    def __init__(self) -> None:
        self.enabled = True
        self.lookup_calls = 0
        self.store_calls = 0
        self.last_lookup_kwargs = None
        self.last_store_kwargs = None

    @staticmethod
    def normalize_question(question: str) -> str:
        return question

    async def lookup(self, **kwargs):
        self.lookup_calls += 1
        self.last_lookup_kwargs = kwargs
        return None

    async def store(self, **kwargs):
        self.store_calls += 1
        self.last_store_kwargs = kwargs


def _trace_events(caplog) -> list[dict]:
    return [
        json.loads(record.message)
        for record in caplog.records
        if record.name == TRACE_LOGGER_NAME
    ]


def test_ask_pipeline_use_cache_false_skips_cache_lookup_and_store():
    embed = _Embed()
    cache = _Cache()
    pipeline = AgentAskPipeline(
        agent_service=_Agent(),
        llm=_LLM(),
        query_refinement_service=_Refiner(),
        embedding_provider=embed,
        semantic_cache_service=cache,
        documents_repository=_DocsRepo(),
    )

    result = asyncio.run(
        pipeline.ask(
            owner_user_id=uuid4(),
            question="q",
            doc_id="doc-1",
            session_id="s1",
            use_cache=False,
        )
    )

    assert result.cache_status == "miss"
    assert cache.lookup_calls == 0
    assert cache.store_calls == 0
    assert embed.calls == 0


def test_ask_pipeline_emits_traces_without_chunk_text(caplog):
    caplog.set_level(logging.INFO, logger=TRACE_LOGGER_NAME)
    embed = _Embed()
    cache = _Cache()
    pipeline = AgentAskPipeline(
        agent_service=_Agent(),
        llm=_LLM(),
        query_refinement_service=_Refiner(),
        embedding_provider=embed,
        semantic_cache_service=cache,
        documents_repository=_DocsRepo(),
    )

    result = asyncio.run(
        pipeline.ask(
            owner_user_id=uuid4(),
            question="q",
            doc_id="doc-1",
            session_id="s1",
            use_cache=True,
            request_id="req-pipeline",
        )
    )

    assert result.cache_status == "miss"
    events = _trace_events(caplog)
    assert [event["event"] for event in events] == [
        "ask.document.checked",
        "ask.query.refined",
        "ask.cache.lookup.started",
        "ask.cache.miss",
        "ask.agent.run.started",
        "ask.agent.run.completed",
        "ask.cache.store.succeeded",
    ]
    assert all(event["request_id"] == "req-pipeline" for event in events)
    assert "question" not in events[1]
    assert "refined_query" not in events[1]
    joined_logs = "\n".join(record.message for record in caplog.records)
    assert "\"snippet\"" not in joined_logs
    assert "\"answer\"" not in joined_logs


def test_ask_pipeline_uses_permission_partitioned_cache_namespace():
    cache = _Cache()
    pipeline = AgentAskPipeline(
        agent_service=_Agent(), llm=_LLM(), query_refinement_service=_Refiner(),
        embedding_provider=_Embed(), semantic_cache_service=cache,
        documents_repository=_DocsRepo(),
    )
    original_pipeline_version = settings.pipeline_version
    owner_id = uuid4()
    try:
        asyncio.run(pipeline.ask(
            owner_user_id=owner_id, question="q", doc_id="doc-1", use_cache=True,
        ))
        first_namespace = cache.last_lookup_kwargs["model_name"]
        assert first_namespace.startswith("model-x:")
        assert len(first_namespace.rsplit(":", 1)[1]) == 64

        settings.pipeline_version = "hybrid-agentic-v2"
        asyncio.run(pipeline.ask(
            owner_user_id=owner_id, question="q", doc_id="doc-1", use_cache=True,
        ))
        second_namespace = cache.last_lookup_kwargs["model_name"]
        assert second_namespace != first_namespace
    finally:
        settings.pipeline_version = original_pipeline_version
