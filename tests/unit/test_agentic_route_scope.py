import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.api.v1.routes.agent import agentic_ask
from src.api.v1.schemas.agent import AgenticAskRequest


class CapturingAgenticService:
    def __init__(self) -> None:
        self.filters = None

    async def run(self, *, question, filters):
        self.filters = filters
        return SimpleNamespace(
            status="rejected", answer="no evidence", citations=[],
            retrieval_attempts=1, failure_reason="no_relevant_documents",
            refined_query=question,
        )


class PermissionStub:
    def __init__(self, *, allowed=(), accessible=None) -> None:
        self.allowed = allowed
        self.accessible = accessible
        self.allowed_kwargs = None

    async def accessible_document(self, **kwargs):
        return self.accessible

    async def allowed_document_ids(self, **kwargs):
        self.allowed_kwargs = kwargs
        return self.allowed


def test_agentic_route_without_doc_id_uses_full_permission_filtered_scope():
    service = CapturingAgenticService()
    permissions = PermissionStub(allowed=("release-3.2", "timeout-sop", "ticket-1842"))
    user = SimpleNamespace(id=uuid4())
    payload = AgenticAskRequest(
        question="Diagnose the 3.2 timeout using release notes, SOP and tickets",
        product_name="Nebula Gateway", product_version="3.2",
        document_types=["release_note", "sop", "ticket"],
        knowledge_space_id="nebula-support",
    )

    response = asyncio.run(agentic_ask(payload, service, permissions, user))

    assert response["status"] == "rejected"
    assert service.filters["allowed_doc_ids"] == [
        "release-3.2", "timeout-sop", "ticket-1842",
    ]
    assert service.filters["deny_all"] is False
    assert permissions.allowed_kwargs["product_version"] == "3.2"


def test_agentic_route_with_no_access_uses_explicit_deny_all_filter():
    service = CapturingAgenticService()
    permissions = PermissionStub(allowed=())

    asyncio.run(agentic_ask(
        AgenticAskRequest(question="unknown"), service, permissions,
        SimpleNamespace(id=uuid4()),
    ))

    assert service.filters["allowed_doc_ids"] == []
    assert service.filters["deny_all"] is True


def test_agentic_exact_doc_scope_rejects_inaccessible_document():
    with pytest.raises(HTTPException) as raised:
        asyncio.run(agentic_ask(
            AgenticAskRequest(question="q", doc_id="private-doc"),
            CapturingAgenticService(), PermissionStub(accessible=None),
            SimpleNamespace(id=uuid4()),
        ))

    assert raised.value.status_code == 404
