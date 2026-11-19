import pytest

from src.settings.agent import AgentSettings


@pytest.mark.parametrize(("value", "expected"), [("", None), ("512", 512)])
def test_agent_max_tokens_from_environment(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: int | None
) -> None:
    monkeypatch.setenv("AGENT_MAX_TOKENS", value)

    assert AgentSettings().agent_max_tokens == expected
