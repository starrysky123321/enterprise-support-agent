from datetime import datetime
from uuid import UUID

from pydantic import BaseModel


class IngestionTaskResponse(BaseModel):
    task_id: UUID
    doc_id: str
    status: str
    progress: int
    retry_count: int
    failure_reason: str | None = None
    content_sha256: str
    created_at: datetime
    updated_at: datetime
    idempotent_reuse: bool = False
