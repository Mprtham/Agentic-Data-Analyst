"""Anthropic API client — wraps the official SDK."""

from __future__ import annotations

from typing import Any

import anthropic

from .base import ContentBlock, LLMClient, LLMResult


class AnthropicClient(LLMClient):
    def __init__(self, api_key: str, model: str, max_tokens: int) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResult:
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            tools=tools,
            messages=messages,
        )

        blocks: list[ContentBlock] = []
        for b in response.content:
            if b.type == "text":
                blocks.append(ContentBlock(type="text", text=b.text))
            elif b.type == "tool_use":
                blocks.append(
                    ContentBlock(
                        type="tool_use",
                        id=b.id,
                        name=b.name,
                        input=dict(b.input),
                    )
                )

        return LLMResult(content=blocks, stop_reason=response.stop_reason)
