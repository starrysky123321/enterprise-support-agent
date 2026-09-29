"""add enterprise document metadata

Revision ID: e7a1b2c3d4e5
Revises: a1b2c3d4e5f6
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e7a1b2c3d4e5"
down_revision: Union[str, Sequence[str], None] = "a1b2c3d4e5f6"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("documents", sa.Column("document_type", sa.String(32), nullable=False, server_default="manual"))
    op.add_column("documents", sa.Column("product_name", sa.String(255), nullable=True))
    op.add_column("documents", sa.Column("product_version", sa.String(128), nullable=True))
    op.add_column("documents", sa.Column("department", sa.String(255), nullable=True))
    op.add_column("documents", sa.Column("knowledge_space_id", sa.String(255), nullable=True))
    op.add_column("documents", sa.Column("visibility", sa.String(32), nullable=False, server_default="private"))
    op.add_column("documents", sa.Column("content_sha256", sa.String(64), nullable=True))
    op.add_column("documents", sa.Column("ingestion_status", sa.String(32), nullable=False, server_default="completed"))
    for column in ("product_name", "product_version", "department", "knowledge_space_id", "content_sha256"):
        op.create_index(f"ix_documents_{column}", "documents", [column])


def downgrade() -> None:
    for column in ("content_sha256", "knowledge_space_id", "department", "product_version", "product_name"):
        op.drop_index(f"ix_documents_{column}", table_name="documents")
    for column in ("ingestion_status", "content_sha256", "visibility", "knowledge_space_id", "department", "product_version", "product_name", "document_type"):
        op.drop_column("documents", column)
