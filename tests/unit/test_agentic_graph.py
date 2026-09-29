import asyncio

from src.agents.langgraph_rag import LangGraphRAGService
from src.infrastructure.llm.local_llm import LocalExtractiveLLM
from src.rag.models import RetrievedChunk
from src.shared.interfaces.llm import LLMResponse


class Retrieval:
    def __init__(self, results):
        self.results = results
        self.calls = 0

    async def retrieve(self, **kwargs):
        self.calls += 1
        return self.results


def test_agentic_graph_rejects_after_bounded_retries():
    retrieval = Retrieval([])
    service = LangGraphRAGService(
        retrieval_service=retrieval, llm=LocalExtractiveLLM(), max_retrieval_attempts=2,
    )
    result = asyncio.run(service.run(question="unknown password", filters={"allowed_doc_ids": ["d"]}))
    assert result.status == "rejected"
    assert result.retrieval_attempts == 2
    assert retrieval.calls == 2
    assert result.citations == []


def test_agentic_graph_maps_citation_to_retrieved_evidence():
    retrieval = Retrieval([
        RetrievedChunk(
            doc_id="d", chunk_id="chunk-7", source="release.md",
            text="CFG-3207 is emitted for the legacy key.", score=0.9, page_number=2,
        )
    ])
    service = LangGraphRAGService(retrieval_service=retrieval, llm=LocalExtractiveLLM())
    result = asyncio.run(service.run(question="What emits CFG-3207?", filters={"allowed_doc_ids": ["d"]}))
    assert result.status == "ok"
    assert "[d:chunk-7]" in result.answer
    assert result.citations[0]["citation_key"] == "d:chunk-7"
    assert result.citations[0]["chunk_id"] == "chunk-7"
    assert result.citations[0]["page_number"] == 2


def test_agentic_graph_keeps_same_chunk_id_unique_across_documents():
    retrieval = Retrieval([
        RetrievedChunk(
            doc_id="release-3.2", chunk_id="chunk-0", source="release.md",
            text="Version 3.2 changed the connection idle timeout default.", score=0.9,
            page_number=1,
        ),
        RetrievedChunk(
            doc_id="ticket-1842", chunk_id="chunk-0", source="ticket.md",
            text="Ticket 1842 resolved timeouts by restoring the idle timeout setting.", score=0.9,
            page_number=3,
        ),
    ])
    service = LangGraphRAGService(retrieval_service=retrieval, llm=LocalExtractiveLLM())

    result = asyncio.run(service.run(
        question="Why did version 3.2 cause connection timeouts?",
        filters={"allowed_doc_ids": ["release-3.2", "ticket-1842"]},
    ))

    assert result.status == "ok"
    assert "[release-3.2:chunk-0]" in result.answer
    assert "[ticket-1842:chunk-0]" in result.answer
    citations = {citation["citation_key"]: citation for citation in result.citations}
    assert citations["release-3.2:chunk-0"]["doc_id"] == "release-3.2"
    assert citations["release-3.2:chunk-0"]["chunk_id"] == "chunk-0"
    assert citations["ticket-1842:chunk-0"]["doc_id"] == "ticket-1842"
    assert citations["ticket-1842:chunk-0"]["chunk_id"] == "chunk-0"


def test_agentic_stream_emits_bounded_stage_events():
    retrieval = Retrieval([
        RetrievedChunk(
            doc_id="d", chunk_id="chunk-1", source="sop.md",
            text="E_CONN_TIMEOUT requires checking idle timeout.", score=0.9,
        )
    ])
    service = LangGraphRAGService(retrieval_service=retrieval, llm=LocalExtractiveLLM())

    async def collect():
        return [event async for event in service.stream(
            question="How to inspect E_CONN_TIMEOUT?", filters={"allowed_doc_ids": ["d"]},
        )]

    events = asyncio.run(collect())
    names = [name for name, _ in events]
    assert names == [
        "retrieval_started", "query_classified", "query_rewritten",
        "documents_retrieved", "answer_generating", "answer_completed",
    ]


def test_agentic_stream_does_not_complete_before_failed_verification_retry():
    class UngroundedLLM:
        model_name = "ungrounded-test"

        async def generate(self, messages, *, config=None, tools=None):
            return LLMResponse(
                content="Unsupported claim [d:chunk-1] [fabricated:chunk-0]", model=self.model_name,
            )

    retrieval = Retrieval([
        RetrievedChunk(
            doc_id="d", chunk_id="chunk-1", source="sop.md",
            text="E_CONN_TIMEOUT requires checking idle timeout.", score=0.9,
        )
    ])
    service = LangGraphRAGService(
        retrieval_service=retrieval, llm=UngroundedLLM(), max_retrieval_attempts=2,
    )

    async def collect():
        return [event async for event in service.stream(
            question="How to inspect E_CONN_TIMEOUT?", filters={"allowed_doc_ids": ["d"]},
        )]

    events = asyncio.run(collect())
    names = [name for name, _ in events]
    assert "answer_completed" not in names
    assert names.count("answer_verification_failed") == 2
    assert names[-1] == "answer_rejected"
