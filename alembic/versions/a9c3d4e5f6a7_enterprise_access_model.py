"""add enterprise access model

Revision ID: a9c3d4e5f6a7
Revises: f8b2c3d4e5f6
"""
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "a9c3d4e5f6a7"
down_revision: Union[str, Sequence[str], None] = "f8b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("user", sa.Column("department", sa.String(255), nullable=True))
    op.create_index("ix_user_department", "user", ["department"])
    op.create_table(
        "knowledge_spaces",
        sa.Column("id", sa.String(255), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("department", sa.String(255), nullable=True),
        sa.Column("created_by", sa.Uuid(), sa.ForeignKey("user.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    op.create_index("ix_knowledge_spaces_department", "knowledge_spaces", ["department"])
    op.create_table(
        "knowledge_space_members",
        sa.Column("knowledge_space_id", sa.String(255), sa.ForeignKey("knowledge_spaces.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("user_id", sa.Uuid(), sa.ForeignKey("user.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("role", sa.String(32), nullable=False, server_default="member"),
    )


def downgrade() -> None:
    op.drop_table("knowledge_space_members")
    op.drop_table("knowledge_spaces")
    op.drop_index("ix_user_department", table_name="user")
    op.drop_column("user", "department")
