from pathlib import Path
from typing import Literal

from pydantic import model_validator
from pydantic_settings import SettingsConfigDict

from src.settings.ai import AISettings
from src.settings.agent import AgentSettings
from src.settings.evaluation import EvaluationSettings
from src.settings.rag import RAGSettings
from src.settings.database import DatabaseSettings
from src.settings.auth import AuthSettings
from src.settings.ingestion import IngestionSettings

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(
    AISettings,
    AgentSettings,
    EvaluationSettings,
    RAGSettings,
    DatabaseSettings,
    AuthSettings,
    IngestionSettings,
):
    environment: Literal["development", "test", "production"] = "development"

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @model_validator(mode="after")
    def reject_unsafe_production_auth(self) -> "Settings":
        """Refuse to start production with placeholder or undersized auth secrets."""
        if self.environment != "production":
            return self

        secret_fields = (
            "access_token_secret_key",
            "refresh_token_secret_key",
            "reset_password_token_secret",
            "verification_token_secret",
        )
        placeholders = {"change_me", "changeme", "secret", "password"}
        secret_values = [getattr(self, field) for field in secret_fields]
        invalid = [
            field
            for field in secret_fields
            if len(getattr(self, field)) < 32 or getattr(self, field).lower() in placeholders
        ]
        if invalid:
            raise ValueError(
                "Production auth secrets must be unique, non-placeholder values of at least "
                f"32 characters: {', '.join(invalid)}"
            )
        if len(set(secret_values)) != len(secret_values):
            raise ValueError("Production auth secrets must use four different values.")
        if not self.auth_cookie_secure:
            raise ValueError("AUTH_COOKIE_SECURE must be true in production.")
        return self


settings = Settings()  # type: ignore[call-arg]
