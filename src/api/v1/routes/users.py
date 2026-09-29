from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database import get_db
from src.modules.users.dependencies import ActiveUserDep
from src.modules.users.dependencies import fastapi_users
from src.modules.users.models import User
from src.modules.users.schemas import DepartmentAssignment, UserRead, UserUpdate

router = APIRouter(tags=["users"])


@router.patch("/users/{user_id}/department", response_model=UserRead)
async def assign_user_department(
    user_id: UUID, payload: DepartmentAssignment, current_user: ActiveUserDep,
    session: AsyncSession = Depends(get_db),
):
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Superuser role required.")
    user = await session.get(User, user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found.")
    user.department = payload.department
    await session.commit()
    await session.refresh(user)
    return user

router.include_router(
    fastapi_users.get_users_router(UserRead, UserUpdate),
    prefix="/users",
    tags=["users"],
)
