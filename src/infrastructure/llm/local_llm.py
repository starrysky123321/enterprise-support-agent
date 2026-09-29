from __future__ import annotations

import ast
from collections.abc import AsyncIterator, Mapping
from typing import Any

from src.shared.interfaces.llm import (
    ChatMessage, GenerationConfig, LLM, LLMResponse, MessageRole, ToolCall,
)


class LocalExtractiveLLM(LLM):
    """Strict local fallback that only answers from retriever tool payloads."""

    @property
    def model_name(self) -> str:
        return "local-extractive-v1"

    @property
    def supports_tool_calls(self) -> bool:
        return True

    async def generate(
        self, messages: list[ChatMessage], *, config: GenerationConfig | None = None,
        tools: list[Mapping[str, Any]] | None = None,
    ) -> LLMResponse:
        tool_message = next((m for m in reversed(messages) if m.role == MessageRole.TOOL), None)
        if tools and tool_message is None:
            query = next((m.content for m in reversed(messages) if m.role == MessageRole.USER), "")
            return LLMResponse(
                content="", model=self.model_name,
                tool_calls=[ToolCall(id="local-retrieve-1", name="retrieve_context", arguments={"query": query})],
            )
        if tool_message is not None:
            try:
                payload = ast.literal_eval(tool_message.content)
                results = payload.get("results", []) if isinstance(payload, dict) else []
            except (SyntaxError, ValueError):
                results = []
            if not results:
                return LLMResponse(
                    content="I could not find the answer in the provided documents.",
                    model=self.model_name,
                )
            bullets = []
            for item in results[:3]:
                text = str(item.get("text", "")).strip().replace("\n", " ")
                chunk_id = str(item.get("chunk_id", ""))
                if text and chunk_id:
                    bullets.append(f"- {text[:360]} [{chunk_id}]")
            return LLMResponse(content="\n".join(bullets), model=self.model_name)

        # Query refinement/judge fallback: retain the original question verbatim.
        content = next((m.content for m in reversed(messages) if m.role == MessageRole.USER), "")
        if "question:" in content:
            content = content.split("question:", 1)[1].split("\n", 1)[0].strip()
        return LLMResponse(content=content.strip(), model=self.model_name)

    async def stream(
        self, messages: list[ChatMessage], *, config: GenerationConfig | None = None,
        tools: list[Mapping[str, Any]] | None = None,
    ) -> AsyncIterator[str]:
        response = await self.generate(messages, config=config, tools=tools)
        yield response.content
