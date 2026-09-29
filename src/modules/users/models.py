from fastapi_users.db import SQLAlchemyBaseUserTableUUID

from sqlalchemy import String
from sqlalchemy.orm import Mapped, mapped_column

from src.infrastructure.database import Base


class User(SQLAlchemyBaseUserTableUUID, Base):
    department: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
