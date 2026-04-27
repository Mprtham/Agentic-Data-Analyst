"""
Report generator — produces HTML and Markdown reports from a completed session.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from ..config import get_settings
from ..engine.session import AnalysisSession
from ..ingestion.schema import DataSchema
from ..logging_config import get_logger
from ..visualization.narrative import NarrativeGenerator

logger = get_logger(__name__)

_TEMPLATE_DIR = Path(__file__).parent


def _basename_filter(path: str) -> str:
    return os.path.basename(path)


class ReportGenerator:
    def __init__(self, output_dir: Path | None = None) -> None:
        cfg = get_settings()
        self._out = output_dir or (cfg.workspace_dir / "reports")
        self._out.mkdir(parents=True, exist_ok=True)
        self._narrative = NarrativeGenerator()
        self._jinja = Environment(
            loader=FileSystemLoader(str(_TEMPLATE_DIR)),
            autoescape=select_autoescape(["html"]),
        )
        self._jinja.filters["basename"] = _basename_filter

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        session: AnalysisSession,
        schema: DataSchema,
        include_narrative: bool = True,
    ) -> tuple[Path, Path]:
        """
        Generate HTML and Markdown reports. Returns (html_path, md_path).
        """
        cfg = get_settings()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        slug = re.sub(r"[^\w]+", "_", session.business_question[:40]).strip("_").lower()
        base_name = f"report_{session.session_id}_{slug}"

        executive_summary = ""
        methodology = ""
        limitations = ""

        if include_narrative and session.findings:
            try:
                executive_summary = self._narrative.executive_summary(
                    session.findings, session.business_question
                )
                limitations = self._narrative.limitations_section(
                    schema.ingest_warnings, []
                )
            except Exception as exc:
                logger.warning("narrative_generation_failed", error=str(exc))

        # Extract agent analysis text from messages
        analysis_narrative = self._extract_analysis_text(session)

        context: dict[str, Any] = {
            "title": f"Analysis Report: {session.business_question[:60]}",
            "session_id": session.session_id,
            "generated_at": now,
            "dataset_source": session.dataset_source,
            "quality_score": round(session.quality_score, 1),
            "business_question": session.business_question,
            "row_count": f"{schema.row_count:,}",
            "col_count": schema.col_count,
            "findings_count": len(session.findings),
            "findings": session.findings,
            "executive_summary": self._md_to_html(executive_summary),
            "analysis_narrative": self._md_to_html(analysis_narrative),
            "quality_warnings": schema.ingest_warnings,
            "artifacts": session.artifacts,
            "methodology": self._md_to_html(methodology),
            "limitations": self._md_to_html(limitations),
            "model": cfg.analyst_model,
        }

        html_path = self._write_html(context, base_name)
        md_path = self._write_markdown(context, base_name)

        logger.info("report_generated", html=str(html_path), md=str(md_path))
        return html_path, md_path

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_html(self, ctx: dict[str, Any], base_name: str) -> Path:
        template = self._jinja.get_template("template.html")
        rendered = template.render(**ctx)
        path = self._out / f"{base_name}.html"
        path.write_text(rendered, encoding="utf-8")
        return path

    def _write_markdown(self, ctx: dict[str, Any], base_name: str) -> Path:
        lines = [
            f"# {ctx['title']}",
            "",
            f"**Session:** {ctx['session_id']}  ",
            f"**Generated:** {ctx['generated_at']}  ",
            f"**Dataset:** {ctx['dataset_source']}  ",
            f"**Quality Score:** {ctx['quality_score']}/100",
            "",
            "## Business Question",
            ctx["business_question"],
            "",
            "## Dataset Overview",
            f"- Rows: {ctx['row_count']}",
            f"- Columns: {ctx['col_count']}",
            "",
        ]

        if ctx.get("findings"):
            lines += ["## Key Findings", ""]
            for f in ctx["findings"]:
                lines.append(f"- {f}")
            lines.append("")

        if ctx.get("quality_warnings"):
            lines += ["## Data Quality Notes", ""]
            for w in ctx["quality_warnings"]:
                lines.append(f"> ⚠️ {w}")
            lines.append("")

        if ctx.get("artifacts"):
            lines += ["## Outputs", ""]
            for a in ctx["artifacts"]:
                lines.append(f"- [{os.path.basename(a)}]({a})")
            lines.append("")

        path = self._out / f"{base_name}.md"
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    @staticmethod
    def _extract_analysis_text(session: AnalysisSession) -> str:
        texts = []
        for msg in session.messages:
            if msg.get("role") == "assistant":
                content = msg.get("content", [])
                if isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and block.get("type") == "text":
                            texts.append(block["text"])
                        elif hasattr(block, "type") and block.type == "text":
                            texts.append(block.text)
                elif isinstance(content, str):
                    texts.append(content)
        return "\n\n---\n\n".join(texts)

    @staticmethod
    def _md_to_html(text: str) -> str:
        """Minimal markdown → HTML (paragraphs and bullets only)."""
        if not text:
            return ""
        try:
            import markdown
            return markdown.markdown(text, extensions=["tables"])
        except ImportError:
            # Very basic fallback
            lines = text.split("\n")
            html_lines = []
            for line in lines:
                if line.startswith("- ") or line.startswith("* "):
                    html_lines.append(f"<li>{line[2:]}</li>")
                elif line.startswith("# "):
                    html_lines.append(f"<h3>{line[2:]}</h3>")
                elif line.strip():
                    html_lines.append(f"<p>{line}</p>")
            return "\n".join(html_lines)
