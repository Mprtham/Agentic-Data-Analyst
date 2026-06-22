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
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

  html, body, [class*="css"] {
    font-family: 'Inter', system-ui, -apple-system, sans-serif;
    line-height: 1.6;
  }

  /* ── Streamlit chrome: hide menu/footer but NEVER hide sidebar toggle ── */
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  .stDeployButton { display: none; }
  div[data-testid="stToolbar"] { display: none; }
  [data-testid="stHeader"] { background: transparent !important; border: none !important; }
  [data-testid="stDecoration"] { display: none !important; }

  /* ── Sidebar toggle: belt-and-braces visibility guarantee ── */
  [data-testid="stSidebarCollapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    background: #1e293b !important;
    border-right: 1px solid #334155 !important;
  }
  [data-testid="stSidebarCollapsedControl"] button {
    color: #94a3b8 !important;
  }
  [data-testid="stSidebarCollapsedControl"] button:hover {
    color: #f1f5f9 !important;
    background: rgba(99,102,241,0.12) !important;
  }

  /* ── App background ── */
  .stApp { background: #0f172a; color: #f1f5f9; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
    background: #1e293b !important;
    border-right: 1px solid #334155;
  }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  [data-testid="stSidebar"] .stMarkdown p { color: #94a3b8 !important; }
  [data-testid="stSidebar"] label {
    color: #cbd5e1 !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
  }
  [data-testid="stSidebar"] hr { border-color: #334155 !important; }

  /* ── File uploader ── */
  [data-testid="stFileUploader"] > div {
    border: 2px dashed #6366f1 !important;
    border-radius: 12px !important;
    background: rgba(99,102,241,0.08) !important;
    transition: all 0.2s ease;
    padding: 1rem !important;
  }
  [data-testid="stFileUploader"] > div:hover {
    border-color: #818cf8 !important;
    background: rgba(99,102,241,0.14) !important;
  }
  [data-testid="stFileUploader"] label {
    color: #e2e8f0 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
  }
  [data-testid="stFileUploader"] small,
  [data-testid="stFileUploader"] span {
    color: #94a3b8 !important;
    font-size: 0.78rem !important;
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
  [data-testid="stTextInput"] label {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
  }

  /* ── Buttons ── */
  .stButton > button {
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    transition: all 0.18s ease !important;
    border: none !important;
  }
  .primary-btn > button {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%) !important;
    color: #ffffff !important;
    padding: 0.65rem 1.2rem !important;
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
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-top: 0.3rem;
  }

  /* ── Hero band (always visible, compact) ── */
  .hero-band {
    padding: 1.8rem 0 1.4rem;
    border-bottom: 1px solid #334155;
    margin-bottom: 0;
  }
  .hero-title {
    font-size: 2.1rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1.1;
    margin: 0 0 0.4rem;
    letter-spacing: -0.03em;
  }
  .hero-title .gr {
    background: linear-gradient(135deg, #818cf8, #6366f1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }
  .hero-tagline {
    font-size: 1rem;
    color: #cbd5e1;
    margin: 0;
    font-weight: 400;
    max-width: 680px;
    line-height: 1.6;
  }

  /* ── Landing page body ── */
  .opening-para {
    font-size: 0.95rem;
    color: #cbd5e1;
    line-height: 1.85;
    max-width: 820px;
    margin: 1.8rem 0 1.6rem;
  }
  .info-col {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 14px;
    padding: 1.3rem 1.4rem 1.4rem;
  }
  .section-label {
    font-size: 0.68rem;
    font-weight: 700;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin: 0 0 0.85rem;
  }
  .check-list {
    list-style: none;
    padding: 0;
    margin: 0;
  }
  .check-item {
    display: flex;
    align-items: flex-start;
    gap: 0.55rem;
    font-size: 0.875rem;
    color: #94a3b8;
    margin-bottom: 0.6rem;
    line-height: 1.5;
  }
  .check-item .tick {
    color: #10b981;
    font-weight: 700;
    flex-shrink: 0;
    font-size: 0.875rem;
    margin-top: 0.02rem;
  }
  .check-item .dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #6366f1;
    flex-shrink: 0;
    margin-top: 0.48rem;
  }
  .outcomes-bar {
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 1rem 1.4rem;
    margin: 1.4rem 0 0;
    font-size: 0.88rem;
    color: #a5b4fc;
    line-height: 1.65;
  }

  /* ── Main-body upload zone ── */
  .upload-zone-header {
    text-align: center;
    padding: 1.8rem 0 0.4rem;
  }
  .upload-zone-icon { font-size: 2.4rem; display: block; margin-bottom: 0.45rem; }
  .upload-zone-title {
    font-size: 1.1rem;
    font-weight: 700;
    color: #f1f5f9;
    margin: 0 0 0.2rem;
  }
  .upload-zone-hint {
    font-size: 0.78rem;
    color: #64748b;
    margin-bottom: 1.1rem;
  }

  /* ── Cold-start notice ── */
  .cold-start-notice {
    background: rgba(245,158,11,0.07);
    border: 1px solid rgba(245,158,11,0.18);
    border-radius: 10px;
    padding: 0.6rem 0.85rem;
    font-size: 0.72rem;
    color: #94a3b8;
    line-height: 1.55;
    margin-top: 1.1rem;
    text-align: left;
  }
  .cold-start-notice strong { color: #f59e0b; font-weight: 600; }

  /* ── Session cards ── */
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
  .session-meta { font-size: 0.7rem; color: #94a3b8; }

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
  .terminal .log-line.info  { color: #818cf8; }
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
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 0.4rem;
    text-align: center;
  }
  .step-connector { flex: 1; height: 2px; background: #334155; margin-top: -18px; }
  .step-connector.done { background: #10b981; }

  /* ── Section headers (used in analysis / report) ── */
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

  /* ── Chart captions ── */
  .chart-caption {
    font-size: 0.72rem;
    color: #94a3b8;
    text-align: center;
    margin-top: 0.4rem;
    font-style: italic;
  }

  /* ── Expanders ── */
  [data-testid="stExpander"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 12px !important;
  }
  [data-testid="stExpander"] summary { color: #e2e8f0 !important; font-weight: 600 !important; }

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

  /* ── Mobile ── */
  @media (max-width: 768px) {
    .hero-title { font-size: 1.5rem !important; }
    .hero-tagline { font-size: 0.88rem !important; }
    .opening-para { font-size: 0.88rem !important; margin: 1.2rem 0 1.2rem !important; }
    .metric-card .metric-value { font-size: 1.3rem !important; }
    .info-col { margin-bottom: 0.75rem; }
    .card { padding: 1rem !important; }
    .upload-zone-header { padding: 1.2rem 0 0.2rem !important; }
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
    <div style="padding:1rem 0 1.2rem;border-bottom:1px solid #334155;margin-bottom:1.4rem;">
      <div style="font-size:1.4rem;margin-bottom:0.3rem;">📊</div>
      <div style="font-size:0.95rem;font-weight:700;color:#f1f5f9;">Agentic Data Analyst</div>
      <div style="font-size:0.7rem;color:#64748b;margin-top:0.1rem;">Powered by Groq / Llama</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;padding:0.3rem 0 0.6rem;">
      <span style="font-size:1.4rem;">☁️</span>
      <div style="font-size:0.85rem;font-weight:600;color:#e2e8f0;margin:0.2rem 0 0.05rem;">Drop your dataset here</div>
      <div style="font-size:0.7rem;color:#64748b;">CSV &nbsp;·&nbsp; Excel &nbsp;·&nbsp; Parquet</div>
    </div>
    """, unsafe_allow_html=True)

    sb_file = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx", "xls", "parquet"],
        label_visibility="collapsed",
        help="Drag and drop, or click to browse",
        key="sb_file",
    )

    st.markdown("<div style='height:0.9rem'></div>", unsafe_allow_html=True)

    sb_q = st.text_input(
        "Business question",
        placeholder="e.g. What drives customer churn?",
        key="sb_q",
    )

    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

    sb_can_run = bool(sb_file and sb_q and sb_q.strip())

    st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
    sb_run_btn = st.button(
        "▶  Run Analysis",
        use_container_width=True,
        disabled=not sb_can_run,
        key="sb_run",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="ghost-btn" style="margin-top:0.5rem">', unsafe_allow_html=True)
    clear_button = st.button("Clear History", use_container_width=True, key="sb_clear")
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
    st.markdown(
        '<p style="font-size:0.7rem;font-weight:600;color:#64748b;text-transform:uppercase;'
        'letter-spacing:0.07em;margin-bottom:0.6rem;">Recent Sessions</p>',
        unsafe_allow_html=True,
    )

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
        st.markdown(
            '<p style="font-size:0.78rem;color:#475569;">No sessions yet.</p>',
            unsafe_allow_html=True,
        )

    st.markdown("""
    <div class="cold-start-notice">
      <strong>⚡ Free tier hosting</strong><br>
      First load may take around 30 seconds to wake. The sidebar toggle arrow is always
      available in the top-left if this panel is closed.
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Hero Band (always visible)
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-band">
  <div class="hero-title">📊 <span class="gr">Agentic</span> Data Analyst</div>
  <p class="hero-tagline">
    An autonomous analyst that turns a raw dataset into an executive-ready report
    in minutes, not days.
  </p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Step Tracker Component
# ─────────────────────────────────────────────────────────────────────────────

def render_step_tracker(active: int):
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
    st.markdown(
        '<div class="section-header"><span>⚙️</span><h2>Running Analysis</h2></div>',
        unsafe_allow_html=True,
    )

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

    quality = state.get("quality_score")
    iters   = state.get("iterations", state.get("findings_count", "—"))
    model   = state.get("model", get_settings().llm_model)
    created = state.get("created_at", "")[:16].replace("T", " ") if state.get("created_at") else "—"

    q_display = f"{float(quality):.0%}" if quality is not None else "—"

    m1, m2, m3, m4 = st.columns(4)
    for col, val, label in [
        (m1, q_display,       "Quality Score"),
        (m2, str(iters),      "Findings"),
        (m3, str(model)[:18], "Model"),
        (m4, created,         "Run At"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
          <div class="metric-value">{val}</div>
          <div class="metric-label">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

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

    charts_dir = session_path / "charts"
    if charts_dir.exists():
        chart_files = sorted(
            list(charts_dir.glob("*.png")) + list(charts_dir.glob("*.html"))
        )
        if chart_files:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="section-header"><span>📈</span><h2>Visualisations</h2></div>',
                unsafe_allow_html=True,
            )
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
                            f'<p class="chart-caption">'
                            f'{chart_file.stem.replace("_", " ").title()}</p>',
                            unsafe_allow_html=True,
                        )
            st.markdown("</div>", unsafe_allow_html=True)

    if tech_appendix.exists():
        with st.expander("📋  Technical Appendix"):
            with open(tech_appendix) as f:
                st.markdown(f.read())

    if state:
        with st.expander("🗂  Session Metadata"):
            st.json(state)

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

def _run_and_report(uploaded_file, question: str):
    file_path = save_uploaded_file(uploaded_file)
    st.session_state.uploaded_file_path = file_path
    run_analysis(file_path, question.strip())
    if st.session_state.analysis_complete and st.session_state.current_session_id:
        session_dir = _find_session_dir(st.session_state.current_session_id)
        if session_dir:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            display_report(session_dir)


if sb_run_btn and sb_can_run:
    _run_and_report(sb_file, sb_q)

elif st.session_state.current_session_id:
    session_dir = _find_session_dir(st.session_state.current_session_id)
    if session_dir:
        display_report(session_dir)
    else:
        st.warning("Session not found.")

else:
    # ── Landing page ─────────────────────────────────────────────────────────

    st.markdown("""
    <p class="opening-para">
      Analysts spend most of their time on repetitive groundwork before any real insight
      appears. Cleaning data, profiling it, checking quality, running the same exploratory
      statistics, and building the same charts from scratch can take hours or days, and
      typically needs someone who knows Python and statistics. For a small team or a
      non-technical manager, getting from a raw CSV to a trustworthy summary is a real
      bottleneck. This tool is designed to remove it. Drop in a dataset, ask a business
      question in plain English, and it can return a structured, quality-checked,
      visualised report automatically.
    </p>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3, gap="medium")

    with c1:
        st.markdown("""
        <div class="info-col">
          <div class="section-label">What it does</div>
          <ul class="check-list">
            <li class="check-item">
              <span class="tick">✓</span>
              Profiles any CSV, Excel or Parquet file automatically: row counts, column types,
              missing values, duplicates, and distributions
            </li>
            <li class="check-item">
              <span class="tick">✓</span>
              Runs data quality checks and flags issues before any analysis is trusted
            </li>
            <li class="check-item">
              <span class="tick">✓</span>
              Performs standard exploratory work: correlations, outliers, trends,
              and segment comparisons
            </li>
            <li class="check-item">
              <span class="tick">✓</span>
              Generates clear visualisations automatically, no chart-building by hand
            </li>
            <li class="check-item">
              <span class="tick">✓</span>
              Answers a plain-English business question about the data
            </li>
            <li class="check-item">
              <span class="tick">✓</span>
              Produces a downloadable, executive-ready report
            </li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="info-col">
          <div class="section-label">The impact it can create</div>
          <ul class="check-list">
            <li class="check-item">
              <span class="dot"></span>
              Designed to compress hours of manual exploratory analysis into minutes
            </li>
            <li class="check-item">
              <span class="dot"></span>
              Can let a non-technical user get a trustworthy data summary without
              writing a line of code
            </li>
            <li class="check-item">
              <span class="dot"></span>
              Frees skilled analysts from repetitive groundwork so they can focus
              on decisions and recommendations
            </li>
            <li class="check-item">
              <span class="dot"></span>
              Reduces the risk of acting on unchecked or poor-quality data
            </li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="info-col">
          <div class="section-label">Who it is for</div>
          <ul class="check-list">
            <li class="check-item">
              <span class="dot"></span>
              Data analysts who want to skip the repetitive first-pass work
            </li>
            <li class="check-item">
              <span class="dot"></span>
              Small teams and founders with data but no dedicated analyst
            </li>
            <li class="check-item">
              <span class="dot"></span>
              Managers and ops or finance staff who need a quick, trustworthy
              read on a dataset
            </li>
            <li class="check-item">
              <span class="dot"></span>
              Anyone who needs a fast, defensible summary before a decision
            </li>
          </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="outcomes-bar">
      Built around the outcomes businesses actually value: faster time-to-insight,
      lower analyst cost per report, fewer decisions made on bad data, and consistent,
      repeatable reporting.
    </div>
    """, unsafe_allow_html=True)

    # ── Main-body upload zone ─────────────────────────────────────────────────
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    _, upload_col, _ = st.columns([1, 2, 1])
    with upload_col:
        st.markdown("""
        <div class="upload-zone-header">
          <span class="upload-zone-icon">☁️</span>
          <div class="upload-zone-title">Drop your dataset here</div>
          <div class="upload-zone-hint">CSV &nbsp;·&nbsp; Excel (.xlsx, .xls) &nbsp;·&nbsp; Parquet</div>
        </div>
        """, unsafe_allow_html=True)

        main_file = st.file_uploader(
            "Choose your file",
            type=["csv", "xlsx", "xls", "parquet"],
            key="main_file",
            help="Drag and drop, or click to browse",
        )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        main_q = st.text_input(
            "Business question",
            placeholder="e.g. What drives customer churn?",
            key="main_q",
        )

        main_can_run = bool(main_file and main_q and main_q.strip())

        st.markdown("<div style='height:0.4rem'></div>", unsafe_allow_html=True)
        st.markdown('<div class="primary-btn">', unsafe_allow_html=True)
        main_run_btn = st.button(
            "▶  Run Analysis",
            use_container_width=True,
            disabled=not main_can_run,
            key="main_run",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="cold-start-notice">
          <strong>⚡ Hosted on a free tier</strong> — the first load may take around
          30 seconds to wake. If the sidebar is closed, tap the arrow in the top-left
          corner to reopen it.
        </div>
        """, unsafe_allow_html=True)

    if main_run_btn and main_can_run:
        _run_and_report(main_file, main_q)
