"""Narrative generation — converts raw findings into polished report text."""

from __future__ import annotations

import anthropic

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)

_NARRATIVE_SYSTEM = """You are a technical writer specialising in data analysis reports.
Convert raw analytical findings into clear, professional prose.
Use UK English spelling. Be precise about confidence levels and limitations.
Never overstate certainty. Keep executive summaries under 100 words."""


class NarrativeGenerator:
    def __init__(self) -> None:
        cfg = get_settings()
        self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.analyst_model

    def executive_summary(self, findings: list[str], question: str) -> str:
        prompt = (
            f"Business question: {question}\n\n"
            f"Raw findings:\n" + "\n".join(f"- {f}" for f in findings) +
            "\n\nWrite a 3-bullet executive summary (no jargon, action-oriented)."
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=400,
            system=_NARRATIVE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def methodology_section(self, methods_used: list[str]) -> str:
        prompt = (
            "Write a concise Methodology section (2–3 short paragraphs) describing "
            "the following analytical techniques used:\n" +
            "\n".join(f"- {m}" for m in methods_used) +
            "\n\nInclude assumptions and why each method was chosen."
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=600,
            system=_NARRATIVE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def limitations_section(self, schema_warnings: list[str], notes: list[str]) -> str:
        items = schema_warnings + notes
        prompt = (
            "Write a Limitations & Caveats section for a data analysis report. "
            "Be honest and specific. Known issues:\n" +
            "\n".join(f"- {i}" for i in items)
        )
        response = self._client.messages.create(
            model=self._model,
            max_tokens=400,
            system=_NARRATIVE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text
