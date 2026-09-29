"""add ingestion tasks

Revision ID: f8b2c3d4e5f6
Revises: e7a1b2c3d4e5
"""
from typing import Sequence, Union
import sqlalchemy as sa
from alembic import op

revision: str = "f8b2c3d4e5f6"
down_revision: Union[str, Sequence[str], None] = "e7a1b2c3d4e5"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "ingestion_tasks",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("doc_id", sa.String(255), sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("owner_user_id", sa.Uuid(), sa.ForeignKey("user.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("progress", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("failure_reason", sa.Text(), nullable=True),
        sa.Column("content_sha256", sa.String(64), nullable=False),
        sa.Column("payload_path", sa.String(1024), nullable=False),
        sa.Column("content_kind", sa.String(16), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )
    for column in ("doc_id", "owner_user_id", "status", "content_sha256"):
        op.create_index(f"ix_ingestion_tasks_{column}", "ingestion_tasks", [column])


def downgrade() -> None:
    op.drop_table("ingestion_tasks")
