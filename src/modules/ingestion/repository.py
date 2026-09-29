from __future__ import annotations

from uuid import UUID

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.ingestion.models import IngestionTask


class IngestionTaskRepository:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def create(
        self, *, owner_user_id: UUID, doc_id: str, content_sha256: str,
        payload_path: str, content_kind: str,
    ) -> IngestionTask:
        task = IngestionTask(
            owner_user_id=owner_user_id, doc_id=doc_id, content_sha256=content_sha256,
            payload_path=payload_path, content_kind=content_kind,
        )
        self._session.add(task)
        await self._session.flush()
        return task

    async def get_owned(self, *, task_id: UUID, owner_user_id: UUID) -> IngestionTask | None:
        result = await self._session.execute(
            select(IngestionTask).where(
                IngestionTask.id == task_id, IngestionTask.owner_user_id == owner_user_id,
            )
        )
        return result.scalar_one_or_none()

    async def find_idempotent(
        self, *, owner_user_id: UUID, content_sha256: str, doc_id: str,
    ) -> IngestionTask | None:
        result = await self._session.execute(
            select(IngestionTask).where(
                IngestionTask.owner_user_id == owner_user_id,
                IngestionTask.content_sha256 == content_sha256,
                IngestionTask.doc_id == doc_id,
            ).order_by(IngestionTask.created_at.desc())
        )
        return result.scalars().first()

    async def update(
        self, *, task_id: UUID, status: str, progress: int,
        failure_reason: str | None = None, increment_retry: bool = False,
    ) -> IngestionTask | None:
        task = await self._session.get(IngestionTask, task_id)
        if task is None:
            return None
        task.status = status
        task.progress = max(0, min(progress, 100))
        task.failure_reason = failure_reason
        if increment_retry:
            task.retry_count += 1
        await self._session.flush()
        return task
