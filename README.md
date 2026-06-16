# Agentic Data Analyst

> A ReAct agent that autonomously profiles datasets, runs statistical tests, and produces revenue-impact models — no human prompting required between steps.

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-sandboxed-2496ED?style=flat-square&logo=docker&logoColor=white)](https://docker.com)
[![ReAct](https://img.shields.io/badge/Pattern-ReAct_Agent-FF6B35?style=flat-square)](https://arxiv.org/abs/2210.03629)

[Case study →](https://portfolio-iota-taupe-34.vercel.app/work/agentic-data-analyst)

---

## What it does

Give the agent a CSV or database connection and a business question. It autonomously:

1. **Profiles** the dataset — dtypes, nulls, distributions, outliers
2. **Plans** which statistical tests are relevant (t-test, chi-square, ANOVA, regression)
3. **Executes** tests in a sandboxed Docker environment
4. **Interprets** results in business terms
5. **Models** revenue or cost impact with confidence intervals

**Key outcomes**
- Reduced analyst time on exploratory analysis by ~70%
- Reproducible analysis audit trail stored as structured JSON
- Safe code execution: all Python runs inside an isolated Docker container

---

## Architecture

```
User prompt
     ↓
ReAct Loop ──── Think ──── Tool call ──── Observe ──── (repeat)
     ↓
Final report (markdown + JSON)
```

**Tool registry (12 tools)**

| Tool | What it does |
|---|---|
| `profile_dataset` | Compute summary stats, missing value map, cardinality |
| `run_statistical_test` | Dispatch to scipy: t-test, chi-square, ANOVA, Mann-Whitney |
| `fit_regression` | OLS / logistic regression with coefficient table |
| `model_revenue_impact` | Project financial impact with bootstrapped CI |
| `plot_distribution` | Generate matplotlib chart, return base64 PNG |
| `write_python` | Write arbitrary Python to sandbox filesystem |
| `execute_python` | Run Python in Docker container, return stdout + stderr |
| … | 5 more utility tools |

---

## Stack

- **Agent framework**: LangChain + custom ReAct loop
- **LLM**: Claude claude-sonnet-4-6 (extended thinking for planning step)
- **Code sandbox**: Docker Python 3.11 slim, resource-limited (256MB RAM, 30s timeout)
- **Statistical engine**: scipy, statsmodels, pingouin
- **Data layer**: pandas, polars, SQLAlchemy (Postgres + SQLite)
- **Output**: Structured JSON + markdown report

---

## Quick start

```bash
git clone https://github.com/Mprtham/agentic-data-analyst
cd agentic-data-analyst
pip install -r requirements.txt
cp .env.example .env   # add ANTHROPIC_API_KEY
docker pull python:3.11-slim   # sandbox image

python main.py --data examples/sales.csv \
               --question "Which product categories are driving churn?"
```

The live demo runs step-by-step in the browser at [portfolio-iota-taupe-34.vercel.app/#lab](https://portfolio-iota-taupe-34.vercel.app/#lab).

---

## Project structure

```
agentic-data-analyst/
├── agent/
│   ├── react_loop.py       # Core ReAct think-act-observe loop
│   ├── tool_registry.py    # Tool definitions + dispatch
│   └── planner.py          # Extended thinking planning step
├── tools/
│   ├── stats.py            # Statistical test implementations
│   ├── sandbox.py          # Docker execution wrapper
│   └── revenue.py          # Impact modelling utilities
├── main.py                 # CLI entrypoint
└── examples/               # Sample datasets + expected outputs
```
