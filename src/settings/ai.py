from typing import Literal, Optional
from pydantic import SecretStr
from pydantic_settings import BaseSettings
from pydantic import Field


class AISettings(BaseSettings):
    llm_provider: Literal["local", "openai", "qwen"] = "local"
    openai_key: Optional[str] = Field(default=None)
    ollama_key: Optional[str] = Field(default=None)
    ollama_base_url: Optional[str]= Field(default=None)
    model: Optional[str] = "gpt-oss:120b-cloud"
    external_request_timeout_s: float = Field(default=30.0, gt=0.0, le=300.0)
    external_max_retries: int = Field(default=2, ge=0, le=5)
    dashscope_api_key: SecretStr | None = Field(default=None)
    dashscope_base_url: str = Field(
        default="https://dashscope.aliyuncs.com/compatible-mode/v1"
    )
    dashscope_model_name: str = Field(default="qwen-plus")
    dashscope_timeout_seconds: float = Field(default=60.0, gt=0.0, le=300.0)
