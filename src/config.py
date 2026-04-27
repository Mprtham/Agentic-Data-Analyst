"""Central configuration.

pydantic-settings maps field names to env vars automatically (case-insensitive).
  llm_api_key  ←→  LLM_API_KEY
  llm_model    ←→  LLM_MODEL
  llm_base_url ←→  LLM_BASE_URL

Supported .env layouts
──────────────────────
  # Groq (default)
  LLM_API_KEY=gsk_...
  LLM_MODEL=llama-3.3-70b-versatile
  LLM_BASE_URL=https://api.groq.com/openai/v1

  # OpenAI
  LLM_API_KEY=sk-...
  LLM_MODEL=gpt-4o
  LLM_BASE_URL=https://api.openai.com/v1

  # Ollama (local, no key required)
  LLM_API_KEY=ollama
  LLM_MODEL=llama3.1:8b
  LLM_BASE_URL=http://localhost:11434/v1
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8-sig",  # handles BOM from Windows text editors
        extra="ignore",
    )

    # ── LLM provider ──────────────────────────────────────────────────
    llm_api_key: str = ""
    llm_model: str = "llama-3.3-70b-versatile"
    llm_base_url: str = "https://api.groq.com/openai/v1"

    # ── Shared ────────────────────────────────────────────────────────
    max_tokens: int = 8192
    workspace_dir: Path = Path("./workspace")
    sandbox_image: str = "analyst-sandbox:latest"
    sandbox_timeout_seconds: int = 30
    log_level: str = "INFO"
    database_url: str | None = None


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()  # type: ignore[call-arg]
    return _settings


def reset_settings() -> None:
    global _settings
    _settings = None
