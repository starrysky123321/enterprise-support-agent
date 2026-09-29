from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings


class IngestionSettings(BaseSettings):
    ingestion_mode: Literal["background", "arq"] = "background"
    ingestion_storage_dir: str = Field(
        default_factory=lambda: str(Path(__file__).resolve().parents[2] / "data" / "uploads")
    )
    redis_url: str = "redis://localhost:6379/0"
    ingestion_max_retries: int = Field(default=2, ge=0, le=5)
