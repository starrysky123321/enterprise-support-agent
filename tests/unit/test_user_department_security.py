import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from src.api.v1.routes.users import assign_user_department
from src.modules.users.schemas import DepartmentAssignment, UserCreate, UserUpdate


def test_public_user_schemas_forbid_department_escalation():
    with pytest.raises(ValidationError):
        UserCreate(
            email="attacker@example.com", password="ChangeMe123!", department="finance",
        )
    with pytest.raises(ValidationError):
        UserUpdate(department="finance")


def test_non_superuser_cannot_assign_department():
    with pytest.raises(HTTPException) as raised:
        asyncio.run(assign_user_department(
            uuid4(), DepartmentAssignment(department="finance"),
            SimpleNamespace(is_superuser=False), SimpleNamespace(),
        ))
    assert raised.value.status_code == 403


def test_superuser_can_assign_department():
    target = SimpleNamespace(department=None)

    class Session:
        committed = False
        refreshed = False

        async def get(self, model, user_id):
            return target

        async def commit(self):
            self.committed = True

        async def refresh(self, user):
            self.refreshed = True

    session = Session()
    result = asyncio.run(assign_user_department(
        uuid4(), DepartmentAssignment(department="finance"),
        SimpleNamespace(is_superuser=True), session,
    ))

    assert result.department == "finance"
    assert session.committed and session.refreshed
