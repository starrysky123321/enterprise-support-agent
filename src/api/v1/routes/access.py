from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database import get_db
from src.modules.access.models import KnowledgeSpace, KnowledgeSpaceMember
from src.modules.users.dependencies import ActiveUserDep

router = APIRouter(prefix="/knowledge-spaces", tags=["knowledge-spaces"])


class SpaceCreate(BaseModel):
    id: str = Field(min_length=1, max_length=255, pattern=r"^[a-zA-Z0-9_-]+$")
    name: str = Field(min_length=1, max_length=255)
    department: str | None = Field(default=None, max_length=255)


class MemberAdd(BaseModel):
    user_id: UUID
    role: str = Field(default="member", pattern=r"^(workspace_admin|member)$")


@router.post("")
async def create_space(
    payload: SpaceCreate, current_user: ActiveUserDep,
    session: AsyncSession = Depends(get_db),
):
    if await session.get(KnowledgeSpace, payload.id):
        raise HTTPException(status_code=409, detail="Knowledge space already exists.")
    space = KnowledgeSpace(
        id=payload.id, name=payload.name, department=payload.department,
        created_by=current_user.id,
    )
    session.add(space)
    session.add(KnowledgeSpaceMember(
        knowledge_space_id=payload.id, user_id=current_user.id, role="workspace_admin",
    ))
    await session.commit()
    return {"id": space.id, "name": space.name, "department": space.department}


@router.post("/{space_id}/members")
async def add_member(
    space_id: str, payload: MemberAdd, current_user: ActiveUserDep,
    session: AsyncSession = Depends(get_db),
):
    result = await session.execute(select(KnowledgeSpaceMember).where(
        KnowledgeSpaceMember.knowledge_space_id == space_id,
        KnowledgeSpaceMember.user_id == current_user.id,
        KnowledgeSpaceMember.role == "workspace_admin",
    ))
    if result.scalar_one_or_none() is None and not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Workspace administrator role required.")
    member = KnowledgeSpaceMember(
        knowledge_space_id=space_id, user_id=payload.user_id, role=payload.role,
    )
    await session.merge(member)
    await session.commit()
    return {"knowledge_space_id": space_id, "user_id": str(payload.user_id), "role": payload.role}
