"""LLM client factory — picks provider from config."""

from __future__ import annotations

from .base import LLMClient


def create_llm_client() -> LLMClient:
    """
    Provider resolution order:
      1. LLM_PROVIDER / LLM_BASE_URL  (unified, preferred)
      2. ANTHROPIC_API_KEY             (legacy)
      3. OLLAMA_BASE_URL               (legacy)
    """
    from ..config import get_settings

    cfg = get_settings()
    provider = cfg.provider

    if provider == "anthropic":
        from .anthropic_client import AnthropicClient

        key = cfg.effective_api_key
        if not key:
            raise RuntimeError(
                "provider=anthropic but no API key found. "
                "Set ANTHROPIC_API_KEY or LLM_API_KEY."
            )
        return AnthropicClient(
            api_key=key,
            model=cfg.effective_model,
            max_tokens=cfg.max_tokens,
        )

    # Any OpenAI-compatible provider: groq, ollama, openai, etc.
    base_url = cfg.effective_base_url
    if not base_url:
        # Ollama default
        if provider == "ollama":
            base_url = "http://localhost:11434/v1"
        else:
            raise RuntimeError(
                f"provider={provider!r} requires LLM_BASE_URL to be set."
            )

    from .openai_compat_client import OpenAICompatClient

    return OpenAICompatClient(
        base_url=base_url,
        model=cfg.effective_model,
        api_key=cfg.effective_api_key,  # None is fine for Ollama
        max_tokens=cfg.max_tokens,
    )
