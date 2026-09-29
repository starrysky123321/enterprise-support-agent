from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.settings.config import Settings


def _production_settings(**overrides: object) -> Settings:
    values: dict[str, object] = {
        "environment": "production",
        "database_url": "sqlite+aiosqlite:///:memory:",
        "access_token_secret_key": "a" * 32,
        "refresh_token_secret_key": "b" * 32,
        "reset_password_token_secret": "c" * 32,
        "verification_token_secret": "d" * 32,
        "auth_cookie_secure": True,
        "_env_file": None,
    }
    values.update(overrides)
    return Settings(**values)  # type: ignore[arg-type]


def test_production_rejects_placeholder_auth_secret() -> None:
    with pytest.raises(ValidationError, match="Production auth secrets"):
        _production_settings(access_token_secret_key="change_me")


def test_production_rejects_reused_auth_secrets() -> None:
    with pytest.raises(ValidationError, match="four different values"):
        _production_settings(refresh_token_secret_key="a" * 32)


def test_production_requires_secure_cookies() -> None:
    with pytest.raises(ValidationError, match="AUTH_COOKIE_SECURE"):
        _production_settings(auth_cookie_secure=False)


def test_production_accepts_distinct_secrets_and_secure_cookies() -> None:
    settings = _production_settings()

    assert settings.environment == "production"
