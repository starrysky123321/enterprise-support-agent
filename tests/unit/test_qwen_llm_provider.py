import pytest
from pydantic import SecretStr

from src.api.v1 import dependencies as deps
from src.settings.config import settings


@pytest.fixture(autouse=True)
def _restore_settings_and_cache():
    fields = [
        "llm_provider", "dashscope_api_key", "dashscope_base_url",
        "dashscope_model_name", "dashscope_timeout_seconds", "external_max_retries",
    ]
    snapshot = {field: getattr(settings, field) for field in fields}
    deps.get_llm.cache_clear()
    try:
        yield
    finally:
        for field, value in snapshot.items():
            setattr(settings, field, value)
        deps.get_llm.cache_clear()


def test_qwen_provider_requires_dashscope_key():
    settings.llm_provider = "qwen"
    settings.dashscope_api_key = None

    with pytest.raises(RuntimeError, match="DASHSCOPE_API_KEY"):
        deps.get_llm()


def test_qwen_provider_uses_openai_compatible_dashscope_configuration(monkeypatch):
    captured = {}

    class FakeOpenAILLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(deps, "OpenAILLM", FakeOpenAILLM)
    settings.llm_provider = "qwen"
    settings.dashscope_api_key = SecretStr("dashscope-test-key")
    settings.dashscope_base_url = "https://dashscope.example/compatible-mode/v1"
    settings.dashscope_model_name = "qwen-plus"
    settings.dashscope_timeout_seconds = 45.0
    settings.external_max_retries = 1

    llm = deps.get_llm()

    assert isinstance(llm, FakeOpenAILLM)
    assert captured == {
        "api_key": "dashscope-test-key",
        "model": "qwen-plus",
        "base_url": "https://dashscope.example/compatible-mode/v1",
        "timeout_s": 45.0,
        "max_retries": 1,
    }
