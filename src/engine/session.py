"""Session state — persists analysis context between turns."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..logging_config import get_logger

logger = get_logger(__name__)


def _serialize_content(content: Any) -> Any:
    """Convert Anthropic SDK content blocks to plain dicts for JSON storage."""
    if isinstance(content, list):
        return [_serialize_content(item) for item in content]
    if isinstance(content, dict):
        return {k: _serialize_content(v) for k, v in content.items()}
    if hasattr(content, "model_dump"):
        return content.model_dump()
    if hasattr(content, "__dict__") and not isinstance(content, (str, int, float, bool)):
        return {
            k: _serialize_content(v)
            for k, v in content.__dict__.items()
            if not k.startswith("_")
        }
    return content


@dataclass
class AnalysisSession:
    session_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    business_question: str = ""
    dataset_source: str = ""
    quality_score: float = 0.0
    messages: list[dict[str, Any]] = field(default_factory=list)
    findings: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    tools_used: list[str] = field(default_factory=list)
    iterations: int = 0
    status: str = "active"

    def add_message(self, role: str, content: Any) -> None:
        self.messages.append({"role": role, "content": _serialize_content(content)})

    def add_finding(self, finding: str) -> None:
        self.findings.append(finding)
        logger.info("finding_recorded", session=self.session_id, finding=finding[:80])

    def add_artifact(self, path: str) -> None:
        if path not in self.artifacts:
            self.artifacts.append(path)

    def record_tool(self, name: str) -> None:
        if name not in self.tools_used:
            self.tools_used.append(name)

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, default=str)
        return path

    @classmethod
    def load(cls, path: Path) -> "AnalysisSession":
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)
