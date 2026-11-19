from __future__ import annotations

import re
from collections.abc import AsyncIterator
from dataclasses import dataclass, replace
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from src.agents.cache_policy import is_no_answer_fallback
from src.infrastructure.llm.local_llm import LOCAL_EXTRACTIVE_MODEL
from src.rag.evidence import EvidenceText, select_local_evidence
from src.rag.models import RetrievedChunk, RetrievalFilter
from src.rag.pipeline import RAGRetrievalService
from src.rag.retrieval.error_codes import find_error_code_definition, lookup_error_code
from src.shared.interfaces.llm import ChatMessage, GenerationConfig, LLM, MessageRole


def citation_key(document: RetrievedChunk) -> str:
    """Return the stable, document-scoped identifier used in generated answers."""
    return f"{document.doc_id}:{document.chunk_id}"


def parse_citation_keys(answer: str) -> set[str]:
    """Extract bracketed citation identifiers from an answer."""
    return set(re.findall(r"\[([^\]]+)\]", answer))


class AgenticRAGState(TypedDict):
    original_query: str
    current_query: str
    messages: list[dict[str, str]]
    filters: dict[str, Any]
    retrieval_attempts: int
    retrieved_documents: list[RetrievedChunk]
    relevant_documents: list[RetrievedChunk]
    answer: str | None
    citations: list[dict[str, Any]]
    groundedness_result: dict[str, Any] | None
    failure_reason: str | None
    query_type: str


@dataclass(slots=True)
class AgenticRAGResult:
    status: str
    answer: str
    citations: list[dict[str, Any]]
    retrieval_attempts: int
    failure_reason: str | None
    refined_query: str


class LangGraphRAGService:
    """Bounded corrective RAG graph; infrastructure stays outside graph state."""

    NO_EVIDENCE = "证据不足，无法基于当前可访问资料给出可靠结论。"

    def __init__(
        self, *, retrieval_service: RAGRetrievalService, llm: LLM,
        max_retrieval_attempts: int = 2, relevance_threshold: float = 0.08,
    ) -> None:
        if not 1 <= max_retrieval_attempts <= 3:
            raise ValueError("max_retrieval_attempts must be between 1 and 3")
        self._retrieval = retrieval_service
        self._llm = llm
        self._max_attempts = max_retrieval_attempts
        self._relevance_threshold = relevance_threshold
        self._graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgenticRAGState)
        graph.add_node("classify_query", self._classify_query)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("retrieve_documents", self._retrieve_documents)
        graph.add_node("grade_documents", self._grade_documents)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("verify_groundedness", self._verify_groundedness)
        graph.add_node("reject_answer", self._reject_answer)
        graph.add_edge(START, "classify_query")
        graph.add_edge("classify_query", "rewrite_query")
        graph.add_edge("rewrite_query", "retrieve_documents")
        graph.add_edge("retrieve_documents", "grade_documents")
        graph.add_conditional_edges(
            "grade_documents", self._after_grade,
            {"generate": "generate_answer", "retry": "rewrite_query", "reject": "reject_answer"},
        )
        graph.add_edge("generate_answer", "verify_groundedness")
        graph.add_conditional_edges(
            "verify_groundedness", self._after_verify,
            {"complete": END, "retry": "rewrite_query", "reject": "reject_answer"},
        )
        graph.add_edge("reject_answer", END)
        return graph.compile()

    async def _classify_query(self, state: AgenticRAGState) -> dict[str, Any]:
        query = state["original_query"].casefold()
        kind = "troubleshooting" if any(word in query for word in ("错误", "超时", "故障", "error", "timeout")) else "knowledge"
        return {"query_type": kind}

    async def _rewrite_query(self, state: AgenticRAGState) -> dict[str, Any]:
        original = state["original_query"].strip()
        attempt = state["retrieval_attempts"]
        if attempt == 0:
            rewritten = original
        else:
            # Expansion never replaces critical error codes, versions or product tokens.
            entities = re.findall(r"\b(?:[A-Z][A-Z0-9_-]{2,}|v?\d+(?:\.\d+)+)\b", original)
            suffix = " 排障 原因 配置 变更 解决方案"
            rewritten = f"{original}{suffix} {' '.join(entities)}".strip()
        return {"current_query": rewritten}

    async def _retrieve_documents(self, state: AgenticRAGState) -> dict[str, Any]:
        raw = state["filters"]
        filters = RetrievalFilter(
            allowed_doc_ids=tuple(raw.get("allowed_doc_ids", ())),
            product_name=raw.get("product_name"), product_version=raw.get("product_version"),
            document_types=tuple(raw.get("document_types", ())),
            knowledge_space_id=raw.get("knowledge_space_id"), department=raw.get("department"),
            deny_all=bool(raw.get("deny_all", False)),
        )
        documents = await self._retrieval.retrieve(
            query=state["current_query"], top_k=8, filters=filters,
        )
        return {
            "retrieved_documents": documents,
            "retrieval_attempts": state["retrieval_attempts"] + 1,
        }

    async def _grade_documents(self, state: AgenticRAGState) -> dict[str, Any]:
        definition = find_error_code_definition(
            lookup_error_code(state["original_query"]), state["retrieved_documents"],
        )
        stop_words = {
            "the", "a", "an", "is", "are", "what", "which", "how", "after", "for",
            "of", "to", "and", "or", "in", "on", "请", "什么", "如何", "的", "和",
        }
        tokens = {
            token for token in re.findall(r"\w+(?:[.-]\w+)*", state["original_query"].casefold())
            if token not in stop_words and len(token) > 1
        }
        relevant: list[RetrievedChunk] = []
        for document in state["retrieved_documents"]:
            document_tokens = set(re.findall(r"\w+(?:[.-]\w+)*", document.text.casefold()))
            overlap = len(tokens & document_tokens) / max(1, len(tokens))
            if document is definition or overlap >= self._relevance_threshold or document.score > 0.75:
                relevant.append(document)
        failure_reason = None if relevant else "no_relevant_documents"
        if relevant and self._llm.model_name == LOCAL_EXTRACTIVE_MODEL:
            relevant = self._extractive_context(state["original_query"], relevant)
            if not relevant:
                failure_reason = "insufficient_evidence"
        return {
            "relevant_documents": relevant,
            "failure_reason": failure_reason,
        }

    def _after_grade(self, state: AgenticRAGState) -> str:
        if state["relevant_documents"]:
            return "generate"
        return "retry" if state["retrieval_attempts"] < self._max_attempts else "reject"

    async def _generate_answer(self, state: AgenticRAGState) -> dict[str, Any]:
        documents = state["relevant_documents"]
        if self._llm.model_name == LOCAL_EXTRACTIVE_MODEL:
            return self._extractive_answer(documents)

        evidence = "\n".join(
            f"[{citation_key(doc)}] {doc.source} page={doc.page_number}: {doc.text}"
            for doc in documents
        )
        messages = [
            ChatMessage(role=MessageRole.SYSTEM, content=(
                "你是企业技术支持助手。只能使用给定证据回答。先给直接结论，再列可能原因和有顺序的排查步骤；"
                "每个事实都附 [doc_id:chunk_id]（例如 [release-3.2:chunk-0]）。"
                "不得添加证据中没有明确出现的环境、机制、原因、参数或操作；"
                "回答前逐项核对问题要求的事实。相关排查建议无法证明具体的版本、配置或数值。"
                f"任何必需事实缺少直接证据时，只返回「{self.NO_EVIDENCE}」，不附引用。"
            )),
            ChatMessage(role=MessageRole.USER, content=f"问题：{state['original_query']}\n证据：\n{evidence}"),
        ]
        try:
            response = await self._llm.generate(
                messages, config=GenerationConfig(temperature=0.0, max_tokens=800, timeout_s=30),
            )
            answer = response.content.strip()
        except Exception:
            answer = ""
        if answer.strip("「」") == self.NO_EVIDENCE or is_no_answer_fallback(answer, exact=True):
            return {"answer": None, "citations": [], "failure_reason": "insufficient_evidence"}
        if not answer:
            extractive = self._extractive_context(state["original_query"], documents)
            if not extractive:
                return {"answer": None, "citations": [], "failure_reason": "insufficient_evidence"}
            return self._extractive_answer(extractive)
        return {"answer": answer, "citations": self._citations(answer, documents)}

    @staticmethod
    def _extractive_context(question: str, documents: list[RetrievedChunk]) -> list[RetrievedChunk]:
        selected = select_local_evidence(
            question,
            [EvidenceText(doc.text, doc.product_version) for doc in documents],
            limit=4,
        )
        return [replace(documents[index], text=text) for index, text in selected]

    def _extractive_answer(self, documents: list[RetrievedChunk]) -> dict[str, Any]:
        evidence_lines = "\n".join(
            f"- {doc.text.strip()[:420]} [{citation_key(doc)}]" for doc in documents
        )
        answer = f"检索到的证据（本地抽取）\n{evidence_lines}"
        return {"answer": answer, "citations": self._citations(answer, documents)}

    @staticmethod
    def _citations(answer: str, documents: list[RetrievedChunk]) -> list[dict[str, Any]]:
        keys = parse_citation_keys(answer)
        return [
            {
                "source": doc.source, "doc_id": doc.doc_id, "chunk_id": doc.chunk_id,
                "citation_key": citation_key(doc),
                "snippet": doc.text[:280], "page_number": doc.page_number,
            }
            for doc in documents
            if citation_key(doc) in keys
        ]

    async def _verify_groundedness(self, state: AgenticRAGState) -> dict[str, Any]:
        valid = {citation_key(doc) for doc in state["relevant_documents"]}
        cited = parse_citation_keys(state["answer"] or "")
        grounded = bool(cited) and cited.issubset(valid)
        return {
            "groundedness_result": {
                "grounded": grounded,
                "cited": sorted(cited),
                "check": "citation_id_integrity_only",
            },
            "failure_reason": None if grounded else state["failure_reason"] or "ungrounded_answer",
            **({"answer": state["answer"], "citations": state["citations"]} if grounded else {}),
        }

    def _after_verify(self, state: AgenticRAGState) -> str:
        if state["groundedness_result"] and state["groundedness_result"].get("grounded"):
            return "complete"
        return "retry" if state["retrieval_attempts"] < self._max_attempts else "reject"

    async def _reject_answer(self, state: AgenticRAGState) -> dict[str, Any]:
        return {"answer": self.NO_EVIDENCE, "citations": []}

    @staticmethod
    def initial_state(*, question: str, filters: dict[str, Any]) -> AgenticRAGState:
        return {
            "original_query": question, "current_query": question, "messages": [],
            "filters": filters, "retrieval_attempts": 0, "retrieved_documents": [],
            "relevant_documents": [], "answer": None, "citations": [],
            "groundedness_result": None, "failure_reason": None, "query_type": "unknown",
        }

    async def run(self, *, question: str, filters: dict[str, Any]) -> AgenticRAGResult:
        final = await self._graph.ainvoke(self.initial_state(question=question, filters=filters))
        rejected = bool(final.get("failure_reason")) and not final.get("citations")
        return AgenticRAGResult(
            status="rejected" if rejected else "ok", answer=final["answer"] or self.NO_EVIDENCE,
            citations=final["citations"], retrieval_attempts=final["retrieval_attempts"],
            failure_reason=final.get("failure_reason"), refined_query=final["current_query"],
        )

    async def stream(self, *, question: str, filters: dict[str, Any]) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        event_names = {
            "classify_query": "query_classified", "rewrite_query": "query_rewritten",
            "retrieve_documents": "documents_retrieved", "generate_answer": "answer_generating",
            "reject_answer": "answer_rejected",
        }
        yield "retrieval_started", {"attempt": 1}
        async for update in self._graph.astream(
            self.initial_state(question=question, filters=filters), stream_mode="updates"
        ):
            for node, values in update.items():
                event = event_names.get(node)
                if node == "verify_groundedness":
                    verification = values.get("groundedness_result") or {}
                    event = (
                        "answer_completed"
                        if verification.get("grounded")
                        else "answer_verification_failed"
                    )
                if event:
                    safe = {key: value for key, value in values.items() if key not in {"retrieved_documents", "relevant_documents"}}
                    if node == "generate_answer":
                        safe.pop("answer", None)
                        safe.pop("citations", None)
                    if node == "retrieve_documents":
                        safe["count"] = len(values.get("retrieved_documents", []))
                    yield event, safe
