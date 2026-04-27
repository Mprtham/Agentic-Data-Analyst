"""Core ReAct analysis loop."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Generator

from openai import OpenAI

from ..config import get_settings
from ..ingestion import DataLoader, DataProfiler, QualityScorer
from ..ingestion.quality import QUALITY_GATE
from ..logging_config import get_logger
from ..tools import ToolRegistry, build_default_registry
from .prompts import SYSTEM_PROMPT, build_analysis_prompt
from .session import AnalysisSession

logger = get_logger(__name__)

_MAX_ITERATIONS = 10
_FINDINGS_RE = re.compile(
    r"---FINDINGS---\s*(.*?)\s*---END FINDINGS---", re.DOTALL
)


def _session_dir(workspace: Path, session_id: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    d = workspace / f"session_{ts}_{session_id}"
    d.mkdir(parents=True, exist_ok=True)
    (d / "charts").mkdir(exist_ok=True)
    return d


class AnalystAgent:
    def __init__(
        self,
        registry: ToolRegistry | None = None,
    ) -> None:
        cfg = get_settings()
        self._client = OpenAI(api_key=cfg.llm_api_key, base_url=cfg.llm_base_url)
        self._messages: list[dict[str, Any]] = []
        self._workspace = cfg.workspace_dir
        self._session: AnalysisSession | None = None
        self._out_dir: Path | None = None
        self._registry: ToolRegistry | None = registry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(
        self,
        source: str,
        question: str,
    ) -> Generator[str, None, None]:
        """Run end-to-end analysis. Yields human-readable status lines."""
        cfg = get_settings()

        # ── 1. Load ──────────────────────────────────────────────────
        yield f"[1/5] Loading data: {source}"
        loader = DataLoader(database_url=cfg.database_url)
        df, source_label = loader.load(source)
        yield f"      Loaded {len(df):,} rows x {len(df.columns)} columns"

        # ── 2. Profile ───────────────────────────────────────────────
        yield "[2/5] Profiling data quality..."
        schema = QualityScorer().score(DataProfiler().profile(df, source_label))
        yield f"      Quality score: {schema.quality_score:.1f}/100"

        # Save quality report regardless of gate result
        quality_md = QualityScorer().report(schema)

        # ── 3. Quality gate ──────────────────────────────────────────
        if schema.quality_score < QUALITY_GATE:
            yield (
                f"[QUALITY GATE FAILED] Score {schema.quality_score:.1f} < {QUALITY_GATE}. "
                "Analysis halted."
            )
            # Write quality report and exit
            out_dir = _session_dir(self._workspace, "halted")
            (out_dir / "quality_report.md").write_text(quality_md, encoding="utf-8")
            yield f"      Quality report saved: {out_dir / 'quality_report.md'}"
            return

        # ── 4. Session + registry ────────────────────────────────────
        session = AnalysisSession()
        session.business_question = question
        session.dataset_source = source_label
        session.quality_score = schema.quality_score
        self._session = session

        out_dir = _session_dir(self._workspace, session.session_id)
        self._out_dir = out_dir

        # Save quality report into session dir
        (out_dir / "quality_report.md").write_text(quality_md, encoding="utf-8")

        # Save the dataset into the session dir so sandbox scripts can read it
        data_path = out_dir / "data.csv"
        df.write_csv(data_path)

        # Build registry pointed at this session's dir (sandbox writes charts here)
        if self._registry is None:
            self._registry = build_default_registry(session_dir=out_dir, data_path=data_path)

        yield f"[3/5] Session {session.session_id} started → {out_dir}"

        # ── 5. ReAct loop ────────────────────────────────────────────
        initial_prompt = build_analysis_prompt(
            business_question=question,
            schema_summary=schema.summary_for_llm(),
            quality_score=schema.quality_score,
        )
        # Tell the agent where the data lives
        initial_prompt += f"\n\nData file path: {data_path}"

        self._messages = [{"role": "user", "content": initial_prompt}]
        session.add_message("user", initial_prompt)
        yield "[4/5] Starting ReAct analysis loop..."

        cfg = get_settings()
        final_text = ""
        for iteration in range(_MAX_ITERATIONS):
            session.iterations += 1
            yield f"      Iteration {iteration + 1}/{_MAX_ITERATIONS}"

            response = self._client.chat.completions.create(
                model=cfg.llm_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self._messages,
                tools=self._registry.list_specs(),
                tool_choice="auto",
                max_tokens=cfg.max_tokens,
            )
            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason
            text = msg.content or ""
            tool_calls = msg.tool_calls or []

            if text:
                final_text = text
                yield f"[AGENT]\n{text}"

            # Store assistant message in OpenAI format
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
            if tool_calls:
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                    }
                    for tc in tool_calls
                ]
            self._messages.append(assistant_msg)
            session.add_message("assistant", assistant_msg)

            # No tool calls → agent finished
            if finish_reason != "tool_calls" or not tool_calls:
                yield "      Agent concluded analysis."
                break

            # Execute tools — one result message per call
            for tc in tool_calls:
                tool_name = tc.function.name
                session.record_tool(tool_name)
                yield f"      -> Tool: {tool_name}"
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    tool_out = self._registry.call(tool_name, **args)
                    tool_out_json = json.dumps(tool_out, default=str)
                    yield f"         Result: {tool_out_json[:200]}..."
                    if isinstance(tool_out, dict):
                        for art in tool_out.get("artifacts", []):
                            session.add_artifact(art)
                except Exception as exc:
                    msg_err = f"{type(exc).__name__}: {exc}"
                    logger.error("tool_error", tool=tool_name, error=msg_err)
                    yield f"         ERROR: {msg_err}"
                    tool_out_json = f"Tool error: {msg_err}"

                tool_result_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_out_json,
                }
                self._messages.append(tool_result_msg)
                session.add_message("tool", tool_result_msg)
        else:
            yield f"      Hit iteration limit ({_MAX_ITERATIONS})."

        session.status = "completed"

        # ── 6. Extract findings from agent's final message ───────────
        findings = _extract_findings(final_text)
        for f in findings:
            session.add_finding(f)

        # Also collect any chart artifacts from charts/ dir
        for p in sorted((out_dir / "charts").glob("*")):
            session.add_artifact(str(p))

        # ── 7. Write report files ─────────────────────────────────────
        yield "[5/5] Writing reports..."
        self._write_reports(session, out_dir, final_text, quality_md)

        yield f"\nDone. Reports saved to: {out_dir}"
        yield f"  executive_summary.md"
        yield f"  technical_appendix.md"
        yield f"  session_state.json"
        if session.artifacts:
            yield f"  charts/ ({len(session.artifacts)} file(s))"

    def chat(self, message: str) -> Generator[str, None, None]:
        """Follow-up question on the active session."""
        if self._session is None or self._registry is None:
            yield "[ERROR] No active session. Run analyse() first."
            return

        cfg = get_settings()
        session = self._session
        self._messages.append({"role": "user", "content": message})
        session.add_message("user", message)

        response = self._client.chat.completions.create(
            model=cfg.llm_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self._messages,
            tools=self._registry.list_specs(),
            tool_choice="auto",
            max_tokens=cfg.max_tokens,
        )
        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        text = msg.content or ""
        tool_calls = msg.tool_calls or []

        if text:
            yield text

        assistant_msg: dict[str, Any] = {"role": "assistant", "content": text}
        if tool_calls:
            assistant_msg["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in tool_calls
            ]
        self._messages.append(assistant_msg)
        session.add_message("assistant", assistant_msg)

        if finish_reason == "tool_calls" and tool_calls:
            for tc in tool_calls:
                yield f"[Tool: {tc.function.name}]"
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    tool_out = self._registry.call(tc.function.name, **args)
                    tool_out_json = json.dumps(tool_out, default=str)
                except Exception as exc:
                    tool_out_json = f"Error: {exc}"
                tool_result_msg: dict[str, Any] = {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_out_json,
                }
                self._messages.append(tool_result_msg)
                session.add_message("tool", tool_result_msg)

            final_resp = self._client.chat.completions.create(
                model=cfg.llm_model,
                messages=[{"role": "system", "content": SYSTEM_PROMPT}] + self._messages,
                max_tokens=cfg.max_tokens,
            )
            final_text = final_resp.choices[0].message.content or ""
            if final_text:
                yield final_text
            final_msg: dict[str, Any] = {"role": "assistant", "content": final_text}
            self._messages.append(final_msg)
            session.add_message("assistant", final_msg)

    def last_session(self) -> AnalysisSession | None:
        return self._session

    def last_output_dir(self) -> Path | None:
        return self._out_dir

    # ------------------------------------------------------------------
    # Report writing
    # ------------------------------------------------------------------

    def _write_reports(
        self,
        session: AnalysisSession,
        out_dir: Path,
        agent_text: str,
        quality_md: str,
    ) -> None:
        # executive_summary.md
        exec_lines = [
            "# Executive Summary",
            "",
            f"**Question:** {session.business_question}",
            f"**Dataset:** {session.dataset_source}",
            f"**Quality score:** {session.quality_score:.1f}/100",
            "",
            "## Key Findings",
            "",
        ]
        if session.findings:
            for f in session.findings:
                exec_lines.append(f"- {f}")
        else:
            exec_lines.append("*(See technical appendix for detailed findings.)*")

        if session.artifacts:
            exec_lines += ["", "## Charts", ""]
            for art in session.artifacts:
                exec_lines.append(f"- [{Path(art).name}]({art})")

        (out_dir / "executive_summary.md").write_text(
            "\n".join(exec_lines), encoding="utf-8"
        )

        # technical_appendix.md
        tech_lines = [
            "# Technical Appendix",
            "",
            f"**Session ID:** {session.session_id}",
            f"**Model:** {get_settings().llm_model}",
            f"**Iterations:** {session.iterations}",
            f"**Tools used:** {', '.join(session.tools_used) or 'none'}",
            "",
            "## Full Agent Analysis",
            "",
            agent_text,
            "",
            "---",
            "",
            quality_md,
        ]
        (out_dir / "technical_appendix.md").write_text(
            "\n".join(tech_lines), encoding="utf-8"
        )

        # session_state.json
        state = {
            "session_id": session.session_id,
            "dataset_path": session.dataset_source,
            "business_question": session.business_question,
            "quality_score": session.quality_score,
            "tools_used": session.tools_used,
            "iterations": session.iterations,
            "findings_count": len(session.findings),
            "artifacts": session.artifacts,
            "status": session.status,
            "created_at": session.created_at,
        }
        (out_dir / "session_state.json").write_text(
            json.dumps(state, indent=2), encoding="utf-8"
        )


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _extract_findings(text: str) -> list[str]:
    """Pull bullet points from the ---FINDINGS--- block."""
    m = _FINDINGS_RE.search(text)
    if not m:
        return []
    block = m.group(1)
    findings = []
    for line in block.splitlines():
        line = line.strip().lstrip("-").strip()
        if line:
            findings.append(line)
    return findings
