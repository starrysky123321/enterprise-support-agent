from typing import Annotated

from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.infrastructure.database import get_db
from src.modules.access.service import PermissionService


def get_permission_service(session: AsyncSession = Depends(get_db)) -> PermissionService:
    return PermissionService(session)


PermissionServiceDep = Annotated[PermissionService, Depends(get_permission_service)]
