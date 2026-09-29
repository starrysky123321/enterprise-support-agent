import asyncio
from uuid import uuid4

import pytest
from arq import Retry
from fastapi import BackgroundTasks
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.infrastructure.database import Base
from src.modules.documents.models import Document
from src.modules.ingestion.models import IngestionTask
from src.modules.ingestion.repository import IngestionTaskRepository
from src.modules.ingestion.worker import arq_process_ingestion
from src.api.v1.routes.ingestion import _dispatch
from src.settings.config import settings
from src.modules.users.models import User


def test_task_idempotency_lookup_and_failure_recovery_state():
    async def run():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            user = User(
                id=uuid4(), email="worker@example.com", hashed_password="x",
                is_active=True, is_superuser=False, is_verified=True,
            )
            session.add(user)
            session.add(Document(id="doc-1", owner_user_id=user.id, ingestion_status="pending"))
            await session.flush()
            repository = IngestionTaskRepository(session)
            task = await repository.create(
                owner_user_id=user.id, doc_id="doc-1", content_sha256="a" * 64,
                payload_path="/tmp/payload", content_kind="text",
            )
            await session.commit()
            duplicate = await repository.find_idempotent(
                owner_user_id=user.id, doc_id="doc-1", content_sha256="a" * 64,
            )
            assert duplicate is not None and duplicate.id == task.id
            await repository.update(
                task_id=task.id, status="failed", progress=40,
                failure_reason="temporary", increment_retry=True,
            )
            await session.commit()
            failed = await session.get(IngestionTask, task.id)
            assert failed is not None
            assert (failed.status, failed.progress, failed.retry_count) == ("failed", 40, 1)
            await repository.update(
                task_id=task.id, status="indexing", progress=65, failure_reason=None,
            )
            await session.commit()
            recovered = await session.get(IngestionTask, task.id)
            assert recovered is not None
            assert recovered.status == "indexing" and recovered.failure_reason is None
        await engine.dispose()

    asyncio.run(run())


def test_arq_handler_converts_processing_failure_to_deferred_retry(monkeypatch):
    async def fail_processing(*args, **kwargs):
        raise FileNotFoundError("payload missing")

    monkeypatch.setattr(
        "src.modules.ingestion.worker.process_ingestion_task", fail_processing,
    )

    with pytest.raises(Retry) as raised:
        asyncio.run(arq_process_ingestion({}, str(uuid4())))

    assert raised.value.defer_score == 1000


def test_arq_dispatch_uses_stable_initial_and_unique_retry_job_ids(monkeypatch):
    calls: list[str] = []
    closed = 0

    class FakePool:
        async def enqueue_job(self, function, task_id, *, _job_id):
            assert function == "arq_process_ingestion"
            assert task_id
            calls.append(_job_id)

        async def aclose(self):
            nonlocal closed
            closed += 1

    async def fake_create_pool(redis_settings):
        return FakePool()

    monkeypatch.setattr("arq.create_pool", fake_create_pool)
    monkeypatch.setattr(settings, "ingestion_mode", "arq")
    task_id = uuid4()

    async def run():
        background = BackgroundTasks()
        await _dispatch(task_id, background)
        await _dispatch(task_id, background, job_id=f"{task_id}-retry-new")

    asyncio.run(run())

    assert calls == [str(task_id), f"{task_id}-retry-new"]
    assert closed == 2
