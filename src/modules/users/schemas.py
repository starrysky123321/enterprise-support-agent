from uuid import UUID

from fastapi_users import schemas
from pydantic import BaseModel, ConfigDict, Field


class UserRead(schemas.BaseUser[UUID]):
    department: str | None = None


class UserCreate(schemas.BaseUserCreate):
    model_config = ConfigDict(extra="forbid")


class UserUpdate(schemas.BaseUserUpdate):
    model_config = ConfigDict(extra="forbid")


class DepartmentAssignment(BaseModel):
    department: str | None = Field(default=None, min_length=1, max_length=255)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
