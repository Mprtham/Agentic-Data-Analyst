"""OpenAI-compatible client — works with Groq, Ollama, OpenAI, and any
provider that exposes a /v1/chat/completions endpoint.

Message translation (stored Anthropic format → OpenAI format) is shared
with the original ollama_client module and lives here.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import httpx

from ..logging_config import get_logger
from .base import ContentBlock, LLMClient, LLMResult

logger = get_logger(__name__)


class OpenAICompatClient(LLMClient):
    """
    Calls  POST {base_url}/chat/completions  with the OpenAI schema.

    api_key is optional — leave None for Ollama (no auth required).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 8192,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._max_tokens = max_tokens
        self._endpoint = f"{self._base_url}/chat/completions"

    def complete(
        self,
        system: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
    ) -> LLMResult:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "system", "content": system}]
            + _to_openai_messages(messages),
            "max_tokens": self._max_tokens,
            "stream": False,
        }
        if tools:
            payload["tools"] = _to_openai_tools(tools)
            payload["tool_choice"] = "auto"

        logger.info(
            "openai_compat_request",
            endpoint=self._endpoint,
            model=self._model,
            n_messages=len(payload["messages"]),
            n_tools=len(tools),
        )

        try:
            resp = httpx.post(
                self._endpoint,
                json=payload,
                headers=headers,
                timeout=120.0,
            )
            resp.raise_for_status()
        except httpx.ConnectError:
            raise RuntimeError(
                f"Cannot connect to {self._base_url}.\n"
                "Check that the service is running and LLM_BASE_URL is correct."
            )
        except httpx.HTTPStatusError as exc:
            raise RuntimeError(
                f"API returned HTTP {exc.response.status_code}: "
                f"{exc.response.text[:400]}"
            )

        data = resp.json()
        choice = data["choices"][0]
        msg = choice["message"]
        finish_reason = choice.get("finish_reason", "stop")

        blocks: list[ContentBlock] = []

        if msg.get("content"):
            blocks.append(ContentBlock(type="text", text=msg["content"]))

        for tc in msg.get("tool_calls") or []:
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {"raw": raw_args}
            else:
                args = raw_args

            blocks.append(
                ContentBlock(
                    type="tool_use",
                    id=tc.get("id") or f"call_{uuid.uuid4().hex[:8]}",
                    name=fn.get("name", ""),
                    input=args,
                )
            )

        stop_reason = (
            "tool_use"
            if finish_reason in ("tool_calls", "tool_use")
            else "end_turn"
        )
        return LLMResult(content=blocks, stop_reason=stop_reason)


# ---------------------------------------------------------------------------
# Message / tool format translators  (Anthropic-stored → OpenAI)
# ---------------------------------------------------------------------------

def _to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Plain string
        if isinstance(content, str):
            out.append({"role": role, "content": content})
            continue

        if not isinstance(content, list) or not content:
            out.append({"role": role, "content": str(content)})
            continue

        first = content[0] if isinstance(content[0], dict) else {}

        # Tool results → role=tool messages (one per result)
        if first.get("type") == "tool_result":
            for tr in content:
                out.append({
                    "role": "tool",
                    "tool_call_id": tr.get("tool_use_id", ""),
                    "content": _coerce_str(tr.get("content", "")),
                })
            continue

        # Mixed content blocks (text + tool_use)
        text_parts = [
            b.get("text", "")
            for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        ]
        tool_uses = [
            b for b in content
            if isinstance(b, dict) and b.get("type") == "tool_use"
        ]

        msg_out: dict[str, Any] = {
            "role": role,
            "content": " ".join(text_parts) if text_parts else None,
        }
        if tool_uses:
            msg_out["tool_calls"] = [
                {
                    "id": tu.get("id", f"call_{uuid.uuid4().hex[:8]}"),
                    "type": "function",
                    "function": {
                        "name": tu.get("name", ""),
                        "arguments": json.dumps(tu.get("input", {})),
                    },
                }
                for tu in tool_uses
            ]
        out.append(msg_out)

    return out


def _to_openai_tools(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in tools
    ]


def _coerce_str(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, default=str)
