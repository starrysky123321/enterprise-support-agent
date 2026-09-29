from pydantic import Field
from pydantic_settings import BaseSettings


class AgentSettings(BaseSettings):
    agent_max_steps: int = Field(default=10, ge=1, le=20)
    agent_temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    agent_max_tokens: int | None = Field(default=None, ge=1)
    agent_timeout_s: float | None = Field(default=30.0, gt=0.0)
    agentic_max_retrieval_attempts: int = Field(default=2, ge=1, le=3)
    prompt_version: str = Field(default="enterprise-support-v1", min_length=1, max_length=64)
    pipeline_version: str = Field(default="hybrid-agentic-v1", min_length=1, max_length=64)
    tracing_enabled: bool = Field(default=True)
    tracing_include_query_text: bool = Field(default=False)
    agent_system_prompt: str = Field(
        default=(
            "You are a strict RAG assistant. You must answer only from retrieved tool outputs.\n\n"
            "Rules:\n"
            "- If retrieval does not explicitly contain the answer, reply exactly: "
            "'I could not find the answer in the provided documents.'\n"
            "- Do not infer, guess, or use prior knowledge.\n"
            "- Do not add details not present in retrieved context.\n"
            "- Every factual bullet must end with its supporting citation in this format: [chunk_id].\n\n"
            "Output format:\n"
            "- Return only bullet points.\n"
            "- Each bullet must end with one citation tag such as [chunk-12].\n"
        )
    )
