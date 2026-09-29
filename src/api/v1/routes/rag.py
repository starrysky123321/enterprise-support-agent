from typing import Annotated
from uuid import uuid4

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from sqlalchemy.exc import IntegrityError

from src.api.v1.dependencies import (
    KeywordIndexDep,
    RAGIngestionServiceDep,
    RAGRetrievalServiceDep,
    VectorStoreDep,
)
from src.api.v1.schemas import (
    RAGIngestPDFResponse,
    RAGIngestTextRequest,
    RAGIngestTextResponse,
    RAGRetrieveRequest,
)
from src.modules.documents import DocumentsRepositoryDep
from src.modules.users.dependencies import ActiveUserDep
from src.settings.config import settings
from src.modules.access.dependencies import PermissionServiceDep
from src.rag.models import RetrievalFilter

router = APIRouter(tags=["rag"])


@router.post("/rag/retrieve")
async def retrieve(
    payload: RAGRetrieveRequest, retrieval_service: RAGRetrievalServiceDep,
    permissions: PermissionServiceDep, current_user: ActiveUserDep,
):
    allowed = await permissions.allowed_document_ids(
        user=current_user, knowledge_space_id=payload.knowledge_space_id,
        product_name=payload.product_name, product_version=payload.product_version,
        document_types=list(payload.document_types),
    )
    if not allowed:
        return {"status": "ok", "results": []}
    results = await retrieval_service.retrieve(
        query=payload.query, top_k=payload.top_k,
        filters=RetrievalFilter(
            allowed_doc_ids=allowed, product_name=payload.product_name,
            product_version=payload.product_version,
            document_types=tuple(payload.document_types),
            knowledge_space_id=payload.knowledge_space_id,
        ),
    )
    return {
        "status": "ok",
        "results": [
            {
                "doc_id": item.doc_id, "chunk_id": item.chunk_id, "source": item.source,
                "score": item.score, "page_number": item.page_number,
                "retrieval_sources": item.retrieval_sources,
            }
            for item in results
        ],
    }


@router.post("/rag/ingest/text", response_model=RAGIngestTextResponse)
async def ingest_text(
    payload: RAGIngestTextRequest,
    ingestion_service: RAGIngestionServiceDep,
    repository: DocumentsRepositoryDep,
    vector_store: VectorStoreDep,
    keyword_index: KeywordIndexDep,
    current_user: ActiveUserDep,
    permissions: PermissionServiceDep,
):
    if not await permissions.can_upload_to_scope(
        user=current_user, visibility=payload.visibility, department=payload.department,
        knowledge_space_id=payload.knowledge_space_id,
    ):
        raise HTTPException(status_code=403, detail="Upload scope is not permitted.")
    resolved_doc_id = payload.doc_id or str(uuid4())
    if await repository.doc_id_exists(doc_id=resolved_doc_id, include_deleted=True):
        raise HTTPException(status_code=409, detail="Document id already exists.")

    try:
        await repository.create_document(
            owner_user_id=current_user.id,
            doc_id=resolved_doc_id,
            source=payload.source or "inline-text",
            document_type=payload.document_type,
            product_name=payload.product_name,
            product_version=payload.product_version,
            department=payload.department,
            knowledge_space_id=payload.knowledge_space_id,
            visibility=payload.visibility,
        )
    except IntegrityError as exc:
        await repository.rollback()
        raise HTTPException(status_code=409, detail="Document id already exists.") from exc

    try:
        result = await ingestion_service.ingest_text(
            text=payload.text,
            source=payload.source,
            doc_id=resolved_doc_id,
            chunking_strategy=payload.chunking_strategy,
            metadata={
                "document_type": payload.document_type,
                "product_name": payload.product_name,
                "product_version": payload.product_version,
                "department": payload.department,
                "knowledge_space_id": payload.knowledge_space_id,
                "visibility": payload.visibility,
            },
        )
        await repository.mark_document_indexed(
            owner_user_id=current_user.id,
            doc_id=resolved_doc_id,
            chunking_strategy=result.chunking_strategy,
            chunk_size=result.chunk_size,
            chunk_overlap=result.chunk_overlap,
        )
        await repository.commit()
    except ValueError as exc:
        await repository.rollback()
        await vector_store.delete_by_doc_id(doc_id=resolved_doc_id)
        await keyword_index.delete_by_doc_id(doc_id=resolved_doc_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        await repository.rollback()
        await vector_store.delete_by_doc_id(doc_id=resolved_doc_id)
        await keyword_index.delete_by_doc_id(doc_id=resolved_doc_id)
        raise HTTPException(status_code=500, detail="Ingestion failed.") from exc

    return RAGIngestTextResponse(
        status="ok",
        doc_id=resolved_doc_id,
        chunks_ingested=result.chunks_ingested,
    )


@router.post("/rag/ingest/pdf", response_model=RAGIngestPDFResponse)
async def ingest_pdf(
    file: Annotated[UploadFile, File(...)],
    ingestion_service: RAGIngestionServiceDep,
    repository: DocumentsRepositoryDep,
    vector_store: VectorStoreDep,
    keyword_index: KeywordIndexDep,
    current_user: ActiveUserDep,
    permissions: PermissionServiceDep,
    source: Annotated[str | None, Form()] = None,
    doc_id: Annotated[str | None, Form()] = None,
    chunking_strategy: Annotated[str | None, Form()] = None,
    document_type: Annotated[str, Form()] = "manual",
    product_name: Annotated[str | None, Form()] = None,
    product_version: Annotated[str | None, Form()] = None,
    department: Annotated[str | None, Form()] = None,
    knowledge_space_id: Annotated[str | None, Form()] = None,
    visibility: Annotated[str, Form()] = "private",
):
    allowed_types = {"manual", "api_doc", "sop", "ticket", "release_note", "faq", "config_doc"}
    if document_type not in allowed_types or visibility not in {"private", "department", "workspace"}:
        raise HTTPException(status_code=422, detail="Invalid document metadata.")
    if not await permissions.can_upload_to_scope(
        user=current_user, visibility=visibility, department=department,
        knowledge_space_id=knowledge_space_id,
    ):
        raise HTTPException(status_code=403, detail="Upload scope is not permitted.")
    if file.content_type not in {"application/pdf", "application/x-pdf"}:
        raise HTTPException(status_code=422, detail="Only PDF files are supported.")

    payload = await file.read()
    max_bytes = settings.rag_pdf_max_mb * 1024 * 1024
    if len(payload) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds max size of {settings.rag_pdf_max_mb} MB.",
        )

    resolved_source = source or file.filename or "uploaded-pdf"
    resolved_doc_id = doc_id or str(uuid4())
    if await repository.doc_id_exists(doc_id=resolved_doc_id, include_deleted=True):
        raise HTTPException(status_code=409, detail="Document id already exists.")
    try:
        await repository.create_document(
            owner_user_id=current_user.id,
            doc_id=resolved_doc_id,
            source=resolved_source,
            document_type=document_type,
            product_name=product_name,
            product_version=product_version,
            department=department,
            knowledge_space_id=knowledge_space_id,
            visibility=visibility,
        )
    except IntegrityError as exc:
        await repository.rollback()
        raise HTTPException(status_code=409, detail="Document id already exists.") from exc

    try:
        result = await ingestion_service.ingest_pdf(
            pdf_bytes=payload,
            source=resolved_source,
            doc_id=resolved_doc_id,
            chunking_strategy=chunking_strategy,
            metadata={
                "document_type": document_type, "product_name": product_name,
                "product_version": product_version, "department": department,
                "knowledge_space_id": knowledge_space_id, "visibility": visibility,
            },
        )
        await repository.mark_document_indexed(
            owner_user_id=current_user.id,
            doc_id=resolved_doc_id,
            chunking_strategy=result.chunking_strategy,
            chunk_size=result.chunk_size,
            chunk_overlap=result.chunk_overlap,
        )
        await repository.commit()
    except ValueError as exc:
        await repository.rollback()
        await vector_store.delete_by_doc_id(doc_id=resolved_doc_id)
        await keyword_index.delete_by_doc_id(doc_id=resolved_doc_id)
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        await repository.rollback()
        await vector_store.delete_by_doc_id(doc_id=resolved_doc_id)
        await keyword_index.delete_by_doc_id(doc_id=resolved_doc_id)
        raise HTTPException(status_code=500, detail="Ingestion failed.") from exc

    return RAGIngestPDFResponse(
        status="ok",
        doc_id=resolved_doc_id,
        chunks_ingested=result.chunks_ingested,
        pages_total=result.pages_total,
        pages_ingested=result.pages_ingested,
        skipped_pages=result.skipped_pages,
        warnings=result.warnings,
    )
