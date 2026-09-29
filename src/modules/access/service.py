from __future__ import annotations

from sqlalchemy import or_, select, true
from sqlalchemy.ext.asyncio import AsyncSession

from src.modules.access.models import KnowledgeSpaceMember
from src.modules.documents.models import Document
from src.modules.users.models import User


class PermissionService:
    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def member_space_ids(self, *, user: User) -> tuple[str, ...]:
        result = await self._session.execute(
            select(KnowledgeSpaceMember.knowledge_space_id).where(
                KnowledgeSpaceMember.user_id == user.id
            )
        )
        return tuple(result.scalars().all())

    async def workspace_role(self, *, user: User, knowledge_space_id: str) -> str | None:
        result = await self._session.execute(
            select(KnowledgeSpaceMember.role).where(
                KnowledgeSpaceMember.user_id == user.id,
                KnowledgeSpaceMember.knowledge_space_id == knowledge_space_id,
            )
        )
        return result.scalar_one_or_none()

    async def can_upload_to_scope(
        self, *, user: User, visibility: str, department: str | None,
        knowledge_space_id: str | None,
    ) -> bool:
        if bool(getattr(user, "is_superuser", False)):
            return True
        if visibility == "private":
            return True
        if visibility == "department":
            return bool(department) and department == getattr(user, "department", None)
        if visibility == "workspace" and knowledge_space_id:
            return await self.workspace_role(
                user=user, knowledge_space_id=knowledge_space_id
            ) == "workspace_admin"
        return False

    async def accessible_document(self, *, user: User, doc_id: str) -> Document | None:
        spaces = await self.member_space_ids(user=user)
        conditions = [Document.owner_user_id == user.id]
        department = getattr(user, "department", None)
        is_admin = bool(getattr(user, "is_superuser", False))
        if department:
            conditions.append(
                (Document.visibility == "department") & (Document.department == department)
            )
        if spaces:
            conditions.append(
                (Document.visibility == "workspace") & Document.knowledge_space_id.in_(spaces)
            )
        stmt = select(Document).where(
            Document.id == doc_id, Document.deleted_at.is_(None),
            true() if is_admin else or_(*conditions),
        )
        return (await self._session.execute(stmt)).scalar_one_or_none()

    async def list_accessible_documents(
        self, *, user: User, limit: int, offset: int,
    ) -> list[Document]:
        spaces = await self.member_space_ids(user=user)
        department = getattr(user, "department", None)
        is_admin = bool(getattr(user, "is_superuser", False))
        conditions = [Document.owner_user_id == user.id]
        if department:
            conditions.append(
                (Document.visibility == "department") & (Document.department == department)
            )
        if spaces:
            conditions.append(
                (Document.visibility == "workspace") & Document.knowledge_space_id.in_(spaces)
            )
        stmt = select(Document).where(Document.deleted_at.is_(None))
        if not is_admin:
            stmt = stmt.where(or_(*conditions))
        result = await self._session.execute(
            stmt.order_by(Document.created_at.desc()).offset(offset).limit(limit)
        )
        return list(result.scalars().all())

    async def can_manage_document(self, *, user: User, document: Document) -> bool:
        if bool(getattr(user, "is_superuser", False)) or document.owner_user_id == user.id:
            return True
        if document.visibility != "workspace" or not document.knowledge_space_id:
            return False
        return await self.workspace_role(
            user=user, knowledge_space_id=document.knowledge_space_id
        ) == "workspace_admin"

    async def allowed_document_ids(
        self, *, user: User, knowledge_space_id: str | None = None,
        product_name: str | None = None, product_version: str | None = None,
        document_types: list[str] | None = None,
    ) -> tuple[str, ...]:
        spaces = await self.member_space_ids(user=user)
        conditions = [Document.owner_user_id == user.id]
        department = getattr(user, "department", None)
        is_admin = bool(getattr(user, "is_superuser", False))
        if department:
            conditions.append(
                (Document.visibility == "department") & (Document.department == department)
            )
        if spaces:
            conditions.append(
                (Document.visibility == "workspace") & Document.knowledge_space_id.in_(spaces)
            )
        stmt = select(Document.id).where(Document.deleted_at.is_(None))
        if not is_admin:
            stmt = stmt.where(or_(*conditions))
        if knowledge_space_id:
            stmt = stmt.where(Document.knowledge_space_id == knowledge_space_id)
        if product_name:
            stmt = stmt.where(Document.product_name == product_name)
        if product_version:
            stmt = stmt.where(Document.product_version == product_version)
        if document_types:
            stmt = stmt.where(Document.document_type.in_(document_types))
        return tuple((await self._session.execute(stmt)).scalars().all())
