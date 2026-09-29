import asyncio
from uuid import uuid4

from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from src.infrastructure.database import Base
from src.modules.access.models import KnowledgeSpace, KnowledgeSpaceMember
from src.modules.access.service import PermissionService
from src.modules.documents.models import Document
from src.modules.users.models import User


def test_cross_department_and_workspace_retrieval_scope():
    async def run():
        engine = create_async_engine("sqlite+aiosqlite:///:memory:")
        async with engine.begin() as connection:
            await connection.run_sync(Base.metadata.create_all)
        factory = async_sessionmaker(engine, expire_on_commit=False)
        async with factory() as session:
            payments = User(
                id=uuid4(), email="payments@example.com", hashed_password="x",
                is_active=True, is_superuser=False, is_verified=True, department="payments",
            )
            sales = User(
                id=uuid4(), email="sales@example.com", hashed_password="x",
                is_active=True, is_superuser=False, is_verified=True, department="sales",
            )
            session.add_all([payments, sales])
            space = KnowledgeSpace(
                id="shared-support", name="Shared Support", created_by=payments.id,
            )
            session.add(space)
            session.add(KnowledgeSpaceMember(
                knowledge_space_id=space.id, user_id=sales.id, role="member",
            ))
            session.add_all([
                Document(id="private-payments", owner_user_id=payments.id, visibility="private"),
                Document(
                    id="department-payments", owner_user_id=payments.id,
                    visibility="department", department="payments",
                ),
                Document(
                    id="workspace-shared", owner_user_id=payments.id,
                    visibility="workspace", knowledge_space_id=space.id,
                ),
            ])
            await session.commit()
            service = PermissionService(session)
            sales_scope = await service.allowed_document_ids(user=sales)
            payments_scope = await service.allowed_document_ids(user=payments)
            assert sales_scope == ("workspace-shared",)
            assert set(payments_scope) == {
                "private-payments", "department-payments", "workspace-shared"
            }
        await engine.dispose()

    asyncio.run(run())
