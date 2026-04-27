# Analyst Agent

**Autonomous Data Analyst AI Agent** — a production-grade system that conducts
end-to-end data analysis independently, with the rigour of a senior human analyst.

---

## Deploy to Render (free)

[![Deploy to Render](https://render.com/images/deploy-to-render-button.svg)](https://render.com/deploy)

**Steps:**

1. **Fork this repo** on GitHub
2. Go to [render.com](https://render.com) → New → **Blueprint**
3. Connect your forked GitHub repo — Render detects `render.yaml` automatically
4. When prompted, set the secret environment variable:
   - `LLM_API_KEY` → your [Groq API key](https://console.groq.com) (free)
5. Click **Apply** — build takes ~3 minutes
6. Your app is live at `https://agentic-data-analyst.onrender.com`

**Free tier notes:**
- Service sleeps after 15 minutes of inactivity — first request after sleep takes ~30 seconds to wake
- 512 MB RAM, 0.1 CPU — sufficient for demo datasets up to ~50 MB
- `workspace/` is ephemeral: sessions are lost on redeploy or sleep cycle
- For persistent sessions, connect a Render Disk ($1/month, 1 GB)

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE LAYER                       │
│              CLI (Typer + Rich)  |  REST API (FastAPI)            │
└────────────────────────┬─────────────────────────────────────────┘
                         │
┌────────────────────────▼─────────────────────────────────────────┐
│                     ORCHESTRATION ENGINE                          │
│          ReAct Loop — Reason → Act → Observe → Iterate            │
│          AnalystAgent  (src/engine/agent.py)                      │
└──────────────┬──────────────────────────────┬────────────────────┘
               │                              │
   ┌───────────▼──────────┐      ┌────────────▼───────────┐
   │   DATA LAYER          │      │   ANALYSIS ENGINE       │
   │  DataLoader           │      │  Claude API (Sonnet)    │
   │  DataProfiler         │      │  Tool Registry          │
   │  QualityScorer        │      │  SandboxExecutor        │
   │  DataSchema           │      │  (Docker / subprocess)  │
   └───────────┬──────────┘      └────────────┬───────────┘
               │                              │
               └──────────────┬───────────────┘
                              │
              ┌───────────────▼──────────────┐
              │   REPORTING & VISUALISATION   │
              │  ReportGenerator (HTML + MD)   │
              │  ChartBuilder (Plotly)         │
              │  NarrativeGenerator            │
              └───────────────┬──────────────┘
                              │
              ┌───────────────▼──────────────┐
              │   MEMORY & STATE STORE        │
              │  MemoryStore (ChromaDB)        │
              │  AnalysisSession (JSON)        │
              └──────────────────────────────┘
```

---

## Autonomy Levels

| Level | Capability | Implemented |
|-------|-----------|-------------|
| L1 | Executes predefined analysis scripts | ✅ |
| L2 | Chooses analysis type from data profile | ✅ |
| L3 | Generates hypotheses, tests them, reports findings | ✅ |
| L4 | Multi-turn dialectic — follow-up questions, iterations | ✅ |

---

## Quick Start

### 1. Install

```bash
cd agent
pip install -e ".[dev]"
```

### 2. Configure

```bash
cp .env.example .env
# Add your ANTHROPIC_API_KEY to .env
```

### 3. Generate sample data

```bash
python data/sample_datasets/generate_samples.py
```

### 4. Run analysis

```bash
# CLI
analyst analyse data/sample_datasets/employee_salary.csv \
  "What factors most strongly predict salary?"

# Profile only
analyst profile data/sample_datasets/customer_churn.csv --save

# List past sessions
analyst sessions
```

### 5. Start API server

```bash
uvicorn src.interface.api:app --reload --port 8000
# Docs at http://localhost:8000/docs
```

### 6. Docker sandbox (recommended for production)

```bash
docker build -f docker/Dockerfile.sandbox -t analyst-sandbox:latest .
```

---

## Tool Registry

The agent has access to these tools via structured tool use:

| Tool | Description |
|------|-------------|
| `python_executor` | Sandboxed code execution (Docker / subprocess) |
| `statistical_test` | t-test, Mann-Whitney, chi-square, ANOVA, Shapiro-Wilk |
| `correlation_matrix` | Pearson / Spearman / Kendall |
| `regression_analysis` | OLS and logistic regression with CIs |
| `time_series_decomp` | STL decomposition into trend + seasonal + residual |
| `clustering` | K-means and hierarchical with silhouette scoring |

---

## Project Structure

```
agent/
├── src/
│   ├── config.py               # Centralised settings (pydantic-settings)
│   ├── logging_config.py       # Structured logging (structlog)
│   ├── ingestion/
│   │   ├── loader.py           # CSV, Excel, Parquet, SQL, REST connectors
│   │   ├── profiler.py         # Auto-profiling — dtype inference, null/outlier detection
│   │   ├── quality.py          # Quality scoring 0–100 + markdown report
│   │   └── schema.py           # DataSchema and ColumnMeta models
│   ├── engine/
│   │   ├── agent.py            # ReAct loop — core reasoning engine
│   │   ├── prompts.py          # System prompt and analysis prompt builder
│   │   └── session.py          # Session persistence (JSON)
│   ├── tools/
│   │   ├── sandbox.py          # Docker/subprocess code execution sandbox
│   │   └── registry.py         # Tool definitions and handlers
│   ├── memory/
│   │   └── store.py            # ChromaDB vector memory
│   ├── visualization/
│   │   ├── charts.py           # Plotly chart builders
│   │   └── narrative.py        # LLM-generated narrative sections
│   ├── reporting/
│   │   ├── generator.py        # HTML + Markdown report builder
│   │   └── template.html       # Jinja2 HTML report template
│   └── interface/
│       ├── cli.py              # Typer CLI
│       └── api.py              # FastAPI REST + SSE streaming
├── tests/
│   ├── test_ingestion.py       # Data loading and profiling tests
│   ├── test_tools.py           # Statistical tools tests
│   └── test_session.py         # Session persistence tests
├── docker/
│   └── Dockerfile.sandbox      # Isolated Python execution environment
├── data/
│   └── sample_datasets/        # Synthetic demo datasets
├── notebooks/
│   └── demo_analysis.ipynb     # End-to-end demo notebook
├── workspace/                  # Runtime outputs (gitignored)
│   ├── analysis_scripts/       # LLM-generated code saved here
│   ├── outputs/                # Charts and artefacts
│   └── reports/                # HTML and MD reports
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Running Tests

```bash
pytest tests/ -v --tb=short
```

---

## Design Decisions

**Why Polars over Pandas?**
Polars is 5–10× faster for in-memory operations and has a cleaner,
expression-based API. Pandas is only used where interoperability requires it
(Excel loading, statsmodels).

**Why Docker sandbox?**
The agent generates and executes arbitrary Python. Running that unsandboxed on
the host is reckless. Docker provides network isolation, filesystem containment,
and resource limits.

**Why separate Claude Code from Git?**
The running agent has no GitHub credentials. It generates artifacts to
`workspace/`; the human owner reviews and commits. This mirrors production
AI governance: the agent proposes, the human disposes.

**Why ReAct over single-shot?**
A single prompt cannot handle the iterative nature of real analysis —
quality checks may fail, code may error, initial hypotheses may need revision.
The ReAct loop lets the agent observe results and adjust, capped at
10 iterations to prevent runaway API spend.

---

## Security Posture

- `ANTHROPIC_API_KEY` is read from environment, never embedded in code
- All LLM-generated code runs in a Docker container with `--network=none`
- No GitHub token or push access in the agent's environment
- Output truncated before injection back into the LLM context

---

## Limitations

- Docker sandbox falls back to in-process execution if Docker is unavailable
  (acceptable for development, not production)
- ChromaDB falls back to keyword matching if not installed
- PDF export requires `weasyprint` which needs system dependencies on some OSes
- No concurrent session support (single-threaded ReAct loop per agent instance)

---

## Roadmap

- [ ] Hypothesis auto-generation from schema (L3 proactive)
- [ ] Anomaly detection module with alert thresholds
- [ ] dbt/Snowflake connector for warehouse-native queries
- [ ] Multi-agent architecture: one agent per dataset, coordinator agent
- [ ] Web dashboard UI (React + Plotly Dash)
