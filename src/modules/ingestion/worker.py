from __future__ import annotations

from pathlib import Path
from uuid import UUID

from src.api.v1.dependencies import get_rag_ingestion_service
from src.infrastructure.database import AsyncSessionFactory
from src.modules.documents.models import Document
from src.modules.ingestion.repository import IngestionTaskRepository
from src.modules.ingestion.models import IngestionTask
from src.settings.config import settings
from arq.connections import RedisSettings
from src.rag.pipeline import IngestionResult, PDFIngestionResult


async def process_ingestion_task(task_id: UUID, *, raise_on_failure: bool = False) -> None:
    async with AsyncSessionFactory() as session:
        repository = IngestionTaskRepository(session)
        task = await session.get(IngestionTask, task_id)
        if task is None or task.status == "completed":
            return
        document = await session.get(Document, task.doc_id)
        if document is None:
            await repository.update(
                task_id=task_id, status="failed", progress=0,
                failure_reason="Document record not found.", increment_retry=True,
            )
            await session.commit()
            if raise_on_failure:
                raise RuntimeError("Document record not found.")
            return
        metadata = {
            "document_type": document.document_type, "product_name": document.product_name,
            "product_version": document.product_version, "department": document.department,
            "knowledge_space_id": document.knowledge_space_id, "visibility": document.visibility,
        }
        try:
            await repository.update(task_id=task_id, status="parsing", progress=15)
            await session.commit()
            import asyncio
            payload = await asyncio.to_thread(Path(task.payload_path).read_bytes)
            await repository.update(task_id=task_id, status="chunking", progress=40)
            await session.commit()
            service = get_rag_ingestion_service()
            await repository.update(task_id=task_id, status="indexing", progress=65)
            await session.commit()
            result: IngestionResult | PDFIngestionResult
            if task.content_kind == "pdf":
                result = await service.ingest_pdf(
                    pdf_bytes=payload, source=document.source, doc_id=document.id, metadata=metadata,
                )
            else:
                result = await service.ingest_text(
                    text=payload.decode("utf-8"), source=document.source,
                    doc_id=document.id, metadata=metadata,
                )
            document.chunking_strategy = result.chunking_strategy
            document.chunk_size = result.chunk_size
            document.chunk_overlap = result.chunk_overlap
            document.ingestion_status = "completed"
            from datetime import datetime, timezone
            document.last_indexed_at = datetime.now(timezone.utc)
            await repository.update(task_id=task_id, status="completed", progress=100)
            await session.commit()
        except Exception as exc:
            await session.rollback()
            task = await session.get(IngestionTask, task_id)
            if task is not None:
                document = await session.get(Document, task.doc_id)
                if document is not None:
                    document.ingestion_status = "failed"
                await repository.update(
                    task_id=task_id, status="failed", progress=task.progress,
                    failure_reason=f"{type(exc).__name__}: {exc}", increment_retry=True,
                )
                await session.commit()
            if raise_on_failure:
                raise


async def arq_process_ingestion(ctx: dict, task_id: str) -> None:
    from arq import Retry

    try:
        await process_ingestion_task(UUID(task_id), raise_on_failure=True)
    except Exception as exc:
        # ARQ enforces WorkerSettings.max_tries; defer prevents a tight retry loop.
        raise Retry(defer=1.0) from exc


class WorkerSettings:
    functions = [arq_process_ingestion]
    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    max_tries = settings.ingestion_max_retries + 1
