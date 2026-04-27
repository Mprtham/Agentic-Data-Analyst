"""Shared types for the LLM client abstraction.

ContentBlock and LLMResult are designed to mirror the Anthropic SDK's
response shape so the agent loop needs no branching per provider.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ContentBlock:
    """One content item returned by the LLM."""

    type: str  # "text" | "tool_use"
    # text blocks
    text: str = ""
    # tool_use blocks
    id: str = ""
    name: str = ""
    input: dict[str, Any] = field(default_factory=dict)


@dataclass
class LLMResult:
    """Normalised response from any LLM provider."""

    content: list[ContentBlock]
    stop_reason: str  # "end_turn" | "tool_use"


class LLMClient:
    """Abstract base — every provider implements complete()."""

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResult:
        raise NotImplementedError
