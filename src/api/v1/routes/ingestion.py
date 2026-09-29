from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from uuid import UUID, uuid4

from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, File, Form, HTTPException, UploadFile
from sqlalchemy.ext.asyncio import AsyncSession

from src.api.v1.schemas.ingestion import IngestionTaskResponse
from src.api.v1.schemas.rag import RAGIngestTextRequest
from src.infrastructure.database import get_db
from src.modules.documents.models import Document
from src.modules.documents.repository import DocumentsRepository
from src.modules.ingestion.models import IngestionTask
from src.modules.ingestion.repository import IngestionTaskRepository
from src.modules.ingestion.worker import process_ingestion_task
from src.modules.users.dependencies import ActiveUserDep
from src.modules.access.dependencies import PermissionServiceDep
from src.settings.config import settings

router = APIRouter(prefix="/ingestion", tags=["ingestion"])


async def _dispatch(
    task_id: UUID, background_tasks: BackgroundTasks, *, job_id: str | None = None,
) -> None:
    if settings.ingestion_mode == "arq":
        try:
            from arq import create_pool
            from arq.connections import RedisSettings
            pool = await create_pool(RedisSettings.from_dsn(settings.redis_url))
            await pool.enqueue_job(
                "arq_process_ingestion", str(task_id), _job_id=job_id or str(task_id),
            )
            await pool.aclose()
        except Exception as exc:
            raise HTTPException(status_code=503, detail=f"Could not enqueue ingestion task: {exc}") from exc
    else:
        background_tasks.add_task(process_ingestion_task, task_id)


async def _dispatch_or_mark_failed(
    *, task: IngestionTask, background_tasks: BackgroundTasks, session: AsyncSession,
    job_id: str | None = None,
) -> None:
    """Persist broker dispatch failures so tasks do not remain pending forever."""
    try:
        await _dispatch(task.id, background_tasks, job_id=job_id)
    except HTTPException as exc:
        document = await session.get(Document, task.doc_id)
        if document is not None:
            document.ingestion_status = "failed"
        await IngestionTaskRepository(session).update(
            task_id=task.id,
            status="failed",
            progress=0,
            failure_reason=f"DispatchError: {exc.detail}",
        )
        await session.commit()
        raise


def _response(task, *, reused: bool = False) -> IngestionTaskResponse:
    return IngestionTaskResponse(
        task_id=task.id, doc_id=task.doc_id, status=task.status, progress=task.progress,
        retry_count=task.retry_count, failure_reason=task.failure_reason,
        content_sha256=task.content_sha256, created_at=task.created_at,
        updated_at=task.updated_at, idempotent_reuse=reused,
    )


@router.post("/text", response_model=IngestionTaskResponse, status_code=202)
async def enqueue_text(
    payload: RAGIngestTextRequest, background_tasks: BackgroundTasks,
    current_user: ActiveUserDep, permissions: PermissionServiceDep,
    session: AsyncSession = Depends(get_db),
):
    if not await permissions.can_upload_to_scope(
        user=current_user, visibility=payload.visibility, department=payload.department,
        knowledge_space_id=payload.knowledge_space_id,
    ):
        raise HTTPException(status_code=403, detail="Upload scope is not permitted.")
    content = payload.text.encode("utf-8")
    digest = hashlib.sha256(content).hexdigest()
    doc_id = payload.doc_id or f"doc-{digest[:24]}"
    task_repository = IngestionTaskRepository(session)
    existing = await task_repository.find_idempotent(
        owner_user_id=current_user.id, content_sha256=digest, doc_id=doc_id,
    )
    if existing is not None:
        return _response(existing, reused=True)
    document_repository = DocumentsRepository(session)
    if await document_repository.doc_id_exists(doc_id=doc_id, include_deleted=True):
        raise HTTPException(status_code=409, detail="Document id already exists with different content.")
    storage = Path(settings.ingestion_storage_dir) / str(current_user.id)
    storage.mkdir(parents=True, exist_ok=True)
    path = storage / f"{digest}.txt"
    await asyncio.to_thread(path.write_bytes, content)
    await document_repository.create_document(
        owner_user_id=current_user.id, doc_id=doc_id, source=payload.source or "inline-text",
        document_type=payload.document_type, product_name=payload.product_name,
        product_version=payload.product_version, department=payload.department,
        knowledge_space_id=payload.knowledge_space_id, visibility=payload.visibility,
        content_sha256=digest, ingestion_status="pending",
    )
    task = await task_repository.create(
        owner_user_id=current_user.id, doc_id=doc_id, content_sha256=digest,
        payload_path=str(path), content_kind="text",
    )
    await session.commit()
    await session.refresh(task)
    await _dispatch_or_mark_failed(
        task=task, background_tasks=background_tasks, session=session,
    )
    return _response(task)


@router.post("/pdf", response_model=IngestionTaskResponse, status_code=202)
async def enqueue_pdf(
    background_tasks: BackgroundTasks, current_user: ActiveUserDep,
    file: Annotated[UploadFile, File(...)],
    permissions: PermissionServiceDep,
    source: Annotated[str | None, Form()] = None,
    doc_id: Annotated[str | None, Form()] = None,
    document_type: Annotated[str, Form()] = "manual",
    product_name: Annotated[str | None, Form()] = None,
    product_version: Annotated[str | None, Form()] = None,
    department: Annotated[str | None, Form()] = None,
    knowledge_space_id: Annotated[str | None, Form()] = None,
    visibility: Annotated[str, Form()] = "private",
    session: AsyncSession = Depends(get_db),
):
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=422, detail="Only PDF files are supported.")
    content = await file.read()
    if len(content) > settings.rag_pdf_max_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail="PDF exceeds configured size limit.")
    if document_type not in {"manual", "api_doc", "sop", "ticket", "release_note", "faq", "config_doc"}:
        raise HTTPException(status_code=422, detail="Invalid document type.")
    if visibility not in {"private", "department", "workspace"}:
        raise HTTPException(status_code=422, detail="Invalid visibility.")
    if not await permissions.can_upload_to_scope(
        user=current_user, visibility=visibility, department=department,
        knowledge_space_id=knowledge_space_id,
    ):
        raise HTTPException(status_code=403, detail="Upload scope is not permitted.")
    digest = hashlib.sha256(content).hexdigest()
    resolved_doc_id = doc_id or f"doc-{digest[:24]}"
    task_repository = IngestionTaskRepository(session)
    existing = await task_repository.find_idempotent(
        owner_user_id=current_user.id, content_sha256=digest, doc_id=resolved_doc_id,
    )
    if existing is not None:
        return _response(existing, reused=True)
    document_repository = DocumentsRepository(session)
    if await document_repository.doc_id_exists(doc_id=resolved_doc_id, include_deleted=True):
        raise HTTPException(status_code=409, detail="Document id already exists with different content.")
    storage = Path(settings.ingestion_storage_dir) / str(current_user.id)
    storage.mkdir(parents=True, exist_ok=True)
    path = storage / f"{digest}.pdf"
    await asyncio.to_thread(path.write_bytes, content)
    await document_repository.create_document(
        owner_user_id=current_user.id, doc_id=resolved_doc_id,
        source=source or file.filename or "uploaded.pdf", document_type=document_type,
        product_name=product_name, product_version=product_version, department=department,
        knowledge_space_id=knowledge_space_id, visibility=visibility,
        content_sha256=digest, ingestion_status="pending",
    )
    task = await task_repository.create(
        owner_user_id=current_user.id, doc_id=resolved_doc_id, content_sha256=digest,
        payload_path=str(path), content_kind="pdf",
    )
    await session.commit()
    await session.refresh(task)
    await _dispatch_or_mark_failed(
        task=task, background_tasks=background_tasks, session=session,
    )
    return _response(task)


@router.get("/tasks/{task_id}", response_model=IngestionTaskResponse)
async def get_task(
    task_id: UUID, current_user: ActiveUserDep,
    session: AsyncSession = Depends(get_db),
):
    task = await IngestionTaskRepository(session).get_owned(
        task_id=task_id, owner_user_id=current_user.id,
    )
    if task is None:
        raise HTTPException(status_code=404, detail="Ingestion task not found.")
    return _response(task)


@router.post("/tasks/{task_id}/retry", response_model=IngestionTaskResponse, status_code=202)
async def retry_task(
    task_id: UUID, background_tasks: BackgroundTasks, current_user: ActiveUserDep,
    session: AsyncSession = Depends(get_db),
):
    repository = IngestionTaskRepository(session)
    task = await repository.get_owned(task_id=task_id, owner_user_id=current_user.id)
    if task is None:
        raise HTTPException(status_code=404, detail="Ingestion task not found.")
    if task.status != "failed":
        raise HTTPException(status_code=409, detail="Only failed tasks can be retried.")
    if not await asyncio.to_thread(Path(task.payload_path).is_file):
        raise HTTPException(
            status_code=409,
            detail="Original payload is unavailable; upload the document again.",
        )
    document = await session.get(Document, task.doc_id)
    if document is None or document.deleted_at is not None:
        raise HTTPException(status_code=409, detail="Document is unavailable for retry.")
    document.ingestion_status = "pending"
    await repository.update(
        task_id=task.id, status="pending", progress=0, failure_reason=None,
    )
    await session.commit()
    # ``updated_at`` is generated by the database on UPDATE and is expired after
    # commit; load it explicitly before constructing the synchronous response model.
    await session.refresh(task)
    await _dispatch_or_mark_failed(
        task=task, background_tasks=background_tasks, session=session,
        job_id=f"{task.id}-retry-{uuid4()}",
    )
    return _response(task)
