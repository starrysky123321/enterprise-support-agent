from time import perf_counter
from uuid import uuid4
import json

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from src.agents import DocumentNotFoundError
from src.api.v1.dependencies import (
    AgentAskPipelineDep,
    LangGraphRAGServiceDep,
)
from src.api.v1.schemas import AgentAskRequest, AgentAskResponse, AgenticAskRequest
from src.api.v1.schemas.agent import AgentCitation as AgentCitationSchema
from src.modules.users.dependencies import ActiveUserDep
from src.modules.access.dependencies import PermissionServiceDep
from src.modules.access.service import PermissionService
from src.modules.users.models import User
from src.shared.tracing import TraceContext, trace_event

router = APIRouter(tags=["agent"])


def _agentic_filters(
    payload: AgenticAskRequest, allowed_doc_ids: tuple[str, ...],
) -> dict:
    return {
        "allowed_doc_ids": list(allowed_doc_ids), "product_name": payload.product_name,
        "product_version": payload.product_version, "document_types": payload.document_types,
        "knowledge_space_id": payload.knowledge_space_id,
        "deny_all": not allowed_doc_ids,
    }


async def _resolve_agentic_scope(
    *, payload: AgenticAskRequest, permissions: PermissionService, current_user: User,
) -> tuple[str, ...]:
    if payload.doc_id is not None:
        document = await permissions.accessible_document(
            user=current_user, doc_id=payload.doc_id,
        )
        if document is None:
            raise HTTPException(status_code=404, detail="Document not found.")
        return (document.id,)
    allowed = await permissions.allowed_document_ids(
        user=current_user,
        knowledge_space_id=payload.knowledge_space_id,
        product_name=payload.product_name,
        product_version=payload.product_version,
        document_types=list(payload.document_types),
    )
    return allowed


@router.post("/agent/ask/agentic")
async def agentic_ask(
    payload: AgenticAskRequest,
    service: LangGraphRAGServiceDep,
    permissions: PermissionServiceDep,
    current_user: ActiveUserDep,
):
    allowed = await _resolve_agentic_scope(
        payload=payload, permissions=permissions, current_user=current_user,
    )
    result = await service.run(
        question=payload.question, filters=_agentic_filters(payload, allowed),
    )
    return {
        "status": result.status, "answer": result.answer, "citations": result.citations,
        "retrieval_attempts": result.retrieval_attempts,
        "failure_reason": result.failure_reason, "refined_query": result.refined_query,
    }


@router.post("/agent/ask/stream")
async def agentic_ask_stream(
    payload: AgenticAskRequest,
    service: LangGraphRAGServiceDep,
    permissions: PermissionServiceDep,
    current_user: ActiveUserDep,
):
    allowed = await _resolve_agentic_scope(
        payload=payload, permissions=permissions, current_user=current_user,
    )

    async def events():
        async for event, data in service.stream(
            question=payload.question, filters=_agentic_filters(payload, allowed),
        ):
            yield f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.post("/agent/ask", response_model=AgentAskResponse)
async def agent_ask(
    request: Request,
    payload: AgentAskRequest,
    ask_pipeline: AgentAskPipelineDep,
    current_user: ActiveUserDep,
    use_cache: bool = Query(default=True),
):
    request_id = getattr(request.state, "request_id", None) or str(uuid4())
    trace_context = TraceContext(
        request_id=request_id,
        doc_id=payload.doc_id,
        owner_user_id=str(current_user.id),
        session_id=payload.session_id,
    )
    started_at = perf_counter()
    trace_event(
        "ask.request.started",
        trace_context=trace_context,
        question=payload.question,
        use_cache=use_cache,
    )
    try:
        result = await ask_pipeline.ask(
            owner_user_id=current_user.id,
            question=payload.question,
            doc_id=payload.doc_id,
            session_id=payload.session_id,
            use_cache=use_cache,
            request_id=request_id,
            current_user=current_user,
        )
    except DocumentNotFoundError as exc:
        trace_event(
            "ask.request.failed",
            trace_context=trace_context,
            question=payload.question,
            error=str(exc),
            status_code=404,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 3),
        )
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        trace_event(
            "ask.request.failed",
            trace_context=trace_context,
            question=payload.question,
            error=str(exc),
            status_code=500,
            elapsed_ms=round((perf_counter() - started_at) * 1000, 3),
        )
        raise

    trace_event(
        "ask.request.succeeded",
        trace_context=trace_context,
        question=payload.question,
        refined_query=result.refined_query,
        cache_status=result.cache_status,
        tools_used=result.tools_used,
        citation_count=len(result.citations),
        elapsed_ms=round((perf_counter() - started_at) * 1000, 3),
    )

    return AgentAskResponse(
        status=result.status,
        cache_status=result.cache_status,  # type: ignore[arg-type]
        refined_query=result.refined_query,
        answer=result.answer,
        steps=result.steps,
        tools_used=result.tools_used,
        citations=[AgentCitationSchema.model_validate(item) for item in result.citations],
    )
