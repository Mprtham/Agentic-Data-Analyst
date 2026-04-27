"""Streamlit frontend for the Agentic Data Analyst."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import json
import shutil
import time
from datetime import datetime

import streamlit as st
from src.interface.streamlit_util import download_dir_as_zip

from src.config import get_settings
from src.engine.agent import AnalystAgent


# ─────────────────────────────────────────────────────────────────────────────
# Page Config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Agentic Data Analyst",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─────────────────────────────────────────────────────────────────────────────
# Global CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
  /* ── Imports ── */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  /* ── Reset & Base ── */
  html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    line-height: 1.6;
  }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }
  .stDeployButton { display: none; }
  div[data-testid="stToolbar"] { display: none; }

  /* ── App background ── */
  .stApp {
    background: #0f172a;
    color: #f1f5f9;
  }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 1px solid #334155;
  }
  [data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
  }
  [data-testid="stSidebar"] .stMarkdown p {
    color: #94a3b8 !important;
  }
  [data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
  }
  [data-testid="stSidebar"] hr {
    border-color: #334155 !important;
  }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] > div {
    border: 2px dashed #6366f1 !important;
    border-radius: 12px !important;
    background: rgba(99, 102, 241, 0.08) !important;
    transition: all 0.2s ease;
    padding: 1.2rem !important;
  }
  [data-testid="stFileUploader"] > div:hover {
    border-color: #818cf8 !important;
    background: rgba(99, 102, 241, 0.14) !important;
  }

  /* ── Text input ── */
  [data-testid="stTextInput"] input {
    background: #0f172a !important;
    border: 1px solid #334155 !important;
    border-radius: 10px !important;
    color: #f1f5f9 !important;
    padding: 0.6rem 0.9rem !important;
    font-size: 0.9rem !important;
    transition: border-color 0.2s;
  }
  [data-testid="stTextInput"] input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.25) !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    transition: all 0.18s ease !important;
    border: none !important;
  }
  /* Primary run button */
  .primary-btn > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
  }
  .primary-btn > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
  }
  .primary-btn > button:disabled {
    background: #334155 !important;
    box-shadow: none !important;
    transform: none !important;
    color: #64748b !important;
  }
  /* Ghost clear button */
  .ghost-btn > button {
    background: transparent !important;
    color: #64748b !important;
    border: 1px solid #334155 !important;
  }
  .ghost-btn > button:hover {
    border-color: #ef4444 !important;
    color: #ef4444 !important;
  }

  /* ── Cards ── */
  .card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: box-shadow 0.2s;
  }
  .card:hover { box-shadow: 0 4px 24px rgba(0,0,0,0.35); }

  .metric-card {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    text-align: center;
  }
  .metric-card .metric-value {
    font-size: 1.75rem;
    font-weight: 700;
    color: #6366f1;
    line-height: 1.2;
  }
  .metric-card .metric-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.3rem;
  }

  /* ── Session cards in sidebar ── */
  .session-card {
    background: #0f172a;
    border: 1px solid #334155;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: border-color 0.2s, background 0.2s;
  }
  .session-card:hover {
    border-color: #6366f1;
    background: rgba(99,102,241,0.08);
  }
  .session-q {
    font-size: 0.82rem;
    font-weight: 500;
    color: #e2e8f0;
    margin-bottom: 0.25rem;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }
  .session-meta {
    font-size: 0.7rem;
    color: #64748b;
  }

  /* ── Badges ── */
  .badge {
    display: inline-block;
    padding: 0.15rem 0.55rem;
    border-radius: 999px;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.04em;
  }
  .badge-green  { background: rgba(16,185,129,0.15); color: #10b981; }
  .badge-orange { background: rgba(245,158,11,0.15);  color: #f59e0b; }
  .badge-red    { background: rgba(239,68,68,0.15);   color: #ef4444; }
  .badge-indigo { background: rgba(99,102,241,0.15);  color: #818cf8; }

  /* ── Terminal log ── */
  .terminal {
    background: #020617;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    font-family: 'JetBrains Mono', 'Fira Code', 'Cascadia Code', monospace;
    font-size: 0.78rem;
    color: #94a3b8;
    max-height: 320px;
    overflow-y: auto;
    line-height: 1.7;
  }
  .terminal .log-line { margin: 0; padding: 0.05rem 0; }
  .terminal .log-line.info  { color: #6366f1; }
  .terminal .log-line.ok    { color: #10b981; }
  .terminal .log-line.warn  { color: #f59e0b; }
  .terminal .log-line.err   { color: #ef4444; }

  /* ── Step tracker ── */
  .step-row {
    display: flex;
    align-items: center;
    gap: 0;
    margin-bottom: 2rem;
  }
  .step-item {
    display: flex;
    flex-direction: column;
    align-items: center;
    flex: 1;
    position: relative;
  }
  .step-circle {
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    font-weight: 700;
    z-index: 1;
  }
  .step-circle.done    { background: #10b981; color: #fff; }
  .step-circle.active  { background: #6366f1; color: #fff; box-shadow: 0 0 0 4px rgba(99,102,241,0.3); }
  .step-circle.pending { background: #1e293b; color: #475569; border: 2px solid #334155; }
  .step-label {
    font-size: 0.68rem;
    font-weight: 600;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.4rem;
    text-align: center;
  }
  .step-connector {
    flex: 1;
    height: 2px;
    background: #334155;
    margin-top: -18px;
  }
  .step-connector.done { background: #10b981; }

  /* ── Section headers ── */
  .section-header {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    margin-bottom: 1.2rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid #334155;
  }
  .section-header h2 {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0;
  }

  /* ── Empty state ── */
  .empty-state {
    text-align: center;
    padding: 5rem 2rem;
  }
  .empty-icon {
    font-size: 4rem;
    margin-bottom: 1.2rem;
    display: block;
  }
  .empty-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f1f5f9;
    margin-bottom: 0.5rem;
  }
  .empty-subtitle {
    font-size: 0.9rem;
    color: #64748b;
    margin-bottom: 2rem;
  }
  .feature-list {
    display: inline-flex;
    flex-direction: column;
    gap: 0.6rem;
    text-align: left;
    margin: 0 auto;
  }
  .feature-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    font-size: 0.85rem;
    color: #94a3b8;
  }
  .feature-dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #6366f1;
    flex-shrink: 0;
  }

  /* ── Chart grid ── */
  .chart-caption {
    font-size: 0.72rem;
    color: #64748b;
    text-align: center;
    margin-top: 0.4rem;
    font-style: italic;
  }

  /* ── Expander overrides ── */
  [data-testid="stExpander"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
  }
  [data-testid="stExpander"] summary {
    color: #e2e8f0 !important;
    font-weight: 600 !important;
  }

  /* ── Progress bar ── */
  [data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #6366f1, #818cf8) !important;
    border-radius: 999px !important;
  }
  [data-testid="stProgressBar"] > div {
    background: #1e293b !important;
    border-radius: 999px !important;
  }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; height: 6px; }
  ::-webkit-scrollbar-track { background: #0f172a; }
  ::-webkit-scrollbar-thumb { background: #334155; border-radius: 999px; }
  ::-webkit-scrollbar-thumb:hover { background: #475569; }

  /* ── Download button ── */
  .dl-btn > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: white !important;
    width: 100% !important;
    padding: 0.7rem !important;
    font-size: 0.9rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
  }
  .dl-btn > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99,102,241,0.45) !important;
  }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Session State
# ─────────────────────────────────────────────────────────────────────────────

if "agent" not in st.session_state:
    st.session_state.agent = AnalystAgent()

if "current_session_id" not in st.session_state:
    st.session_state.current_session_id: str | None = None

if "analysis_output" not in st.session_state:
    st.session_state.analysis_output: list[str] = []

if "analysis_complete" not in st.session_state:
    st.session_state.analysis_complete = False

if "uploaded_file_path" not in st.session_state:
    st.session_state.uploaded_file_path: Path | None = None

if "active_step" not in st.session_state:
    st.session_state.active_step = 0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

STEPS = ["Ingest", "Profile", "Analyse", "Report"]

def _step_index_from_line(line: str) -> int:
    line_lower = line.lower()
    if any(k in line_lower for k in ("ingest", "load", "reading", "parsing")):
        return 1
    if any(k in line_lower for k in ("profile", "quality", "schema")):
        return 2
    if any(k in line_lower for k in ("analys", "finding", "insight", "model", "stat")):
        return 3
    if any(k in line_lower for k in ("report", "summary", "writing", "complete", "done")):
        return 4
    return st.session_state.active_step


def _line_class(line: str) -> str:
    low = line.lower()
    if any(k in low for k in ("error", "fail", "exception")):
        return "err"
    if any(k in low for k in ("warn", "caution")):
        return "warn"
    if any(k in low for k in ("complete", "done", "success", "✓", "✅")):
        return "ok"
    return "info"


def _score_badge(score) -> str:
    try:
        s = float(score)
    except (TypeError, ValueError):
        return '<span class="badge badge-indigo">N/A</span>'
    if s >= 0.8:
        return f'<span class="badge badge-green">{s:.0%}</span>'
    if s >= 0.5:
        return f'<span class="badge badge-orange">{s:.0%}</span>'
    return f'<span class="badge badge-red">{s:.0%}</span>'


def list_past_sessions() -> list[dict]:
    cfg = get_settings()
    workspace = cfg.workspace_dir
    if not workspace.exists():
        return []
    sessions = []
    for session_dir in sorted(workspace.glob("session_*"), reverse=True):
        state_file = session_dir / "session_state.json"
        if state_file.exists():
            try:
                with open(state_file) as f:
                    state = json.load(f)
                sessions.append({
                    "id": state.get("session_id", ""),
                    "path": session_dir,
                    "question": state.get("business_question", "N/A")[:60],
                    "dataset": state.get("dataset_path", "Unknown"),
                    "created_at": state.get("created_at", ""),
                    "quality_score": state.get("quality_score"),
                    "status": state.get("status", ""),
                })
            except (json.JSONDecodeError, IOError):
                pass
    return sessions


def save_uploaded_file(uploaded_file) -> Path:
    cfg = get_settings()
    workspace = cfg.workspace_dir
    workspace.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = workspace / f"uploaded_{timestamp}_{uploaded_file.name}"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


def _find_session_dir(session_id: str) -> Path | None:
    cfg = get_settings()
    if not cfg.workspace_dir.exists():
        return None
    for d in cfg.workspace_dir.glob("session_*"):
        sf = d / "session_state.json"
        if sf.exists():
            try:
                with open(sf) as f:
                    if json.load(f).get("session_id") == session_id:
                        return d
            except (json.JSONDecodeError, IOError):
                pass
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 1.2rem; border-bottom:1px solid #334155; margin-bottom:1.4rem;">
      <div style="font-size:1.5rem; margin-bottom:0.3rem;">📊</div>
      <div style="font-size:1rem; font-weight:700; color:#f1f5f9;">Agentic Data Analyst</div>
      <div style="font-size:0.72rem; color:#64748b; margin-top:0.1rem;">Powered by Claude</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.4rem;">Dataset</p>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Upload CSV, Excel, or Parquet",
        type=["csv", "xlsx", "xls", "parquet"],
        label_visibility="collapsed",
        help="Drop a file or click to browse",
    )
    if not uploaded_file:
        st.markdown('<p style="font-size:0.73rem;color:#475569;margin-top:-0.4rem;">CSV · Excel · Parquet</p>', unsafe_allow_html=True)

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.4rem;">Analysis Question</p>', unsafe_allow_html=True)
    business_question = st.text_input(
        "Question",
        placeholder="e.g. What drives customer churn?",
        label_visibility="collapsed",
    )

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    can_run = bool(uploaded_file and business_question)

    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    run_button = st.button("▶  Run Analysis", use_container_width=True, disabled=not can_run)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="ghost-btn" style="margin-top:0.5rem">', unsafe_allow_html=True)
    clear_button = st.button("Clear History", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if clear_button:
        cfg = get_settings()
        if cfg.workspace_dir.exists():
            shutil.rmtree(cfg.workspace_dir)
        st.session_state.current_session_id = None
        st.session_state.analysis_output = []
        st.session_state.analysis_complete = False
        st.rerun()

    st.markdown("<hr style='border-color:#334155;margin:1.4rem 0 1rem'>", unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.72rem;font-weight:600;color:#64748b;text-transform:uppercase;letter-spacing:0.07em;margin-bottom:0.6rem;">Recent Sessions</p>', unsafe_allow_html=True)

    past_sessions = list_past_sessions()
    if past_sessions:
        for s in past_sessions[:8]:
            date_str = s["created_at"][:10] if s["created_at"] else ""
            badge_html = _score_badge(s["quality_score"])
            col_a, col_b = st.columns([4, 1])
            with col_a:
                st.markdown(f"""
                <div class="session-card">
                  <div class="session-q">{s['question']}</div>
                  <div class="session-meta">{date_str} &nbsp;{badge_html}</div>
                </div>
                """, unsafe_allow_html=True)
            with col_b:
                if st.button("→", key=f"load_{s['id']}", help="Load this session"):
                    st.session_state.current_session_id = s["id"]
                    st.session_state.analysis_complete = True
                    st.rerun()
    else:
        st.markdown('<p style="font-size:0.78rem;color:#475569;">No sessions yet.</p>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding:2rem 0 1.2rem;">
  <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem;">
    <span style="font-size:2.2rem;">📊</span>
    <div>
      <h1 style="margin:0;font-size:1.75rem;font-weight:800;color:#f1f5f9;line-height:1.2;">
        Agentic Data Analyst
      </h1>
      <p style="margin:0;font-size:0.875rem;color:#64748b;margin-top:0.2rem;">
        Autonomous AI-powered data analysis &nbsp;·&nbsp; Powered by Claude
      </p>
    </div>
  </div>
  <hr style="border:none;border-top:1px solid #334155;margin-top:1rem;">
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step Tracker Component
# ─────────────────────────────────────────────────────────────────────────────

def render_step_tracker(active: int):
    """Render 4-step progress indicator. active is 0-based current step index."""
    items = []
    for i, label in enumerate(STEPS):
        if i < active:
            circle = '<div class="step-circle done">✓</div>'
        elif i == active:
            circle = f'<div class="step-circle active">{i+1}</div>'
        else:
            circle = f'<div class="step-circle pending">{i+1}</div>'
        items.append(f'<div class="step-item">{circle}<div class="step-label">{label}</div></div>')

    connectors = []
    for i in range(len(STEPS) - 1):
        cls = "step-connector done" if i < active else "step-connector"
        connectors.append(f'<div class="{cls}"></div>')

    interleaved = []
    for i, item in enumerate(items):
        interleaved.append(item)
        if i < len(connectors):
            interleaved.append(connectors[i])

    st.markdown(
        f'<div class="step-row">{"".join(interleaved)}</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Analysis Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_analysis(data_path: Path, question: str):
    st.session_state.analysis_output = []
    st.session_state.analysis_complete = False
    st.session_state.active_step = 0

    agent = st.session_state.agent

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header"><span>⚙️</span><h2>Running Analysis</h2></div>', unsafe_allow_html=True)

    step_placeholder = st.empty()
    progress_bar = st.progress(0)
    log_placeholder = st.empty()

    log_lines: list[str] = []

    for output_line in agent.analyse(str(data_path), question):
        log_lines.append(output_line)
        st.session_state.analysis_output.append(output_line)

        new_step = _step_index_from_line(output_line)
        if new_step > st.session_state.active_step:
            st.session_state.active_step = new_step

        with step_placeholder.container():
            render_step_tracker(min(st.session_state.active_step, len(STEPS) - 1))

        progress_bar.progress(min(st.session_state.active_step / len(STEPS), 1.0))

        rendered = "".join(
            f'<p class="log-line {_line_class(l)}">&gt; {l}</p>'
            for l in log_lines[-30:]
        )
        log_placeholder.markdown(
            f'<div class="terminal">{rendered}</div>',
            unsafe_allow_html=True,
        )
        time.sleep(0.01)

    # Final state
    with step_placeholder.container():
        render_step_tracker(len(STEPS))
    progress_bar.progress(1.0)

    st.markdown("</div>", unsafe_allow_html=True)
    st.session_state.analysis_complete = True
    last = agent.last_session()
    st.session_state.current_session_id = last.session_id if last else None


# ─────────────────────────────────────────────────────────────────────────────
# Report Renderer
# ─────────────────────────────────────────────────────────────────────────────

def display_report(session_path: Path):
    exec_summary = session_path / "executive_summary.md"
    tech_appendix = session_path / "technical_appendix.md"
    state_file    = session_path / "session_state.json"

    state: dict = {}
    if state_file.exists():
        try:
            with open(state_file) as f:
                state = json.load(f)
        except (json.JSONDecodeError, IOError):
            pass

    # ── Metric cards ────────────────────────────────────────────────────────
    quality   = state.get("quality_score")
    iters     = state.get("iterations", state.get("findings_count", "—"))
    model     = state.get("model", get_settings().analyst_model if hasattr(get_settings(), "analyst_model") else "Claude")
    created   = state.get("created_at", "")[:16].replace("T", " ") if state.get("created_at") else "—"

    q_display = f"{float(quality):.0%}" if quality is not None else "—"

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, q_display, "Quality Score"),
        (m2, str(iters), "Findings"),
        (m3, str(model)[:12], "Model"),
        (m4, created, "Run At"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    # ── Executive summary ────────────────────────────────────────────────────
    if exec_summary.exists():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("""
        <div class="section-header">
          <span>📋</span><h2>Executive Summary</h2>
          <span class="badge badge-indigo" style="margin-left:auto;">Report</span>
        </div>
        """, unsafe_allow_html=True)
        with open(exec_summary) as f:
            st.markdown(f.read())
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Charts ───────────────────────────────────────────────────────────────
    charts_dir = session_path / "charts"
    if charts_dir.exists():
        chart_files = sorted(
            list(charts_dir.glob("*.png")) + list(charts_dir.glob("*.html"))
        )
        if chart_files:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<div class="section-header"><span>📈</span><h2>Visualisations</h2></div>', unsafe_allow_html=True)

            pairs = [chart_files[i:i+2] for i in range(0, len(chart_files), 2)]
            for pair in pairs:
                cols = st.columns(len(pair))
                for col, chart_file in zip(cols, pair):
                    with col:
                        if chart_file.suffix == ".png":
                            st.image(str(chart_file))
                        else:
                            with open(chart_file) as f:
                                st.components.v1.html(f.read(), height=420)
                        st.markdown(
                            f'<p class="chart-caption">{chart_file.stem.replace("_", " ").title()}</p>',
                            unsafe_allow_html=True,
                        )
            st.markdown("</div>", unsafe_allow_html=True)

    # ── Technical appendix ───────────────────────────────────────────────────
    if tech_appendix.exists():
        with st.expander("📋  Technical Appendix"):
            with open(tech_appendix) as f:
                st.markdown(f.read())

    # ── Session metadata ─────────────────────────────────────────────────────
    if state:
        with st.expander("🗂  Session Metadata"):
            st.json(state)

    # ── Download ─────────────────────────────────────────────────────────────
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    zip_buffer = download_dir_as_zip(session_path)
    st.markdown('<div class="dl-btn">', unsafe_allow_html=True)
    st.download_button(
        label="📦  Download Full Report (ZIP)",
        data=zip_buffer,
        file_name=f"{session_path.name}_report.zip",
        mime="application/zip",
        use_container_width=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main Panel
# ─────────────────────────────────────────────────────────────────────────────

if run_button and uploaded_file and business_question:
    file_path = save_uploaded_file(uploaded_file)
    st.session_state.uploaded_file_path = file_path
    run_analysis(file_path, business_question)

    if st.session_state.analysis_complete and st.session_state.current_session_id:
        session_dir = _find_session_dir(st.session_state.current_session_id)
        if session_dir:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            display_report(session_dir)

elif st.session_state.current_session_id:
    session_dir = _find_session_dir(st.session_state.current_session_id)
    if session_dir:
        display_report(session_dir)
    else:
        st.warning("Session not found.")

else:
    # ── Empty state ──────────────────────────────────────────────────────────
    st.markdown("""
    <div class="empty-state">
      <span class="empty-icon">🔍</span>
      <div class="empty-title">Upload a dataset to begin</div>
      <div class="empty-subtitle">
        Drop any CSV, Excel, or Parquet file in the sidebar and ask a question.<br>
        The agent will analyse it end-to-end and return an executive-ready report.
      </div>
      <div class="feature-list">
        <div class="feature-item"><div class="feature-dot"></div>Automatic data quality checks &amp; profiling</div>
        <div class="feature-item"><div class="feature-dot"></div>Statistical rigour — correlation, outliers, trends</div>
        <div class="feature-item"><div class="feature-dot"></div>Visualisations generated automatically</div>
        <div class="feature-item"><div class="feature-dot"></div>Executive summary + technical appendix</div>
        <div class="feature-item"><div class="feature-dot"></div>Full report download as ZIP</div>
      </div>
    </div>
    """, unsafe_allow_html=True)
