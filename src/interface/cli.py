"""Analyst Agent CLI entry point.

Usage:
    analyst analyse data/sample_datasets/employee_salary.csv "What drives salary?"
    analyst profile data/sample_datasets/employee_salary.csv
    analyst chat "Can you break down salary by department?"
"""

from __future__ import annotations

import io
import sys
from pathlib import Path
from typing import Optional

# Force UTF-8 output on Windows to prevent cp1252 encoding errors
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

app = typer.Typer(
    name="analyst",
    help="Autonomous Data Analyst Agent",
    add_completion=False,
)
console = Console(highlight=False)

_agent = None


def _get_agent():
    global _agent
    if _agent is None:
        from ..engine import AnalystAgent
        _agent = AnalystAgent()
    return _agent


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command()
def analyse(
    source: str = typer.Argument(..., help="Path to CSV/Parquet/Excel, or sql:QUERY"),
    question: str = typer.Argument(..., help="Business question to answer"),
) -> None:
    """Run end-to-end autonomous analysis and write reports to workspace/."""
    console.print(Panel.fit(
        f"[bold blue]Analyst Agent[/bold blue]\n[dim]{question}[/dim]",
        border_style="blue",
    ))

    agent = _get_agent()

    try:
        for event in agent.analyse(source, question):
            if event.startswith("[AGENT]"):
                body = event[len("[AGENT]"):].strip()
                if body:
                    console.print(Markdown(body))
            elif event.startswith("[QUALITY GATE FAILED]"):
                console.print(f"[bold red]{event}[/bold red]")
            elif event.startswith("["):
                console.print(f"[dim]{event}[/dim]")
            else:
                console.print(event)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted.[/yellow]")
        raise typer.Exit(0)
    except Exception as exc:
        console.print(f"[bold red]Error: {exc}[/bold red]")
        raise typer.Exit(1)


@app.command()
def chat(
    message: str = typer.Argument(..., help="Follow-up question"),
) -> None:
    """Continue the active session with a follow-up question."""
    agent = _get_agent()
    if agent.last_session() is None:
        console.print("[red]No active session. Run 'analyst analyse' first.[/red]")
        raise typer.Exit(1)

    for event in agent.chat(message):
        console.print(Markdown(event) if len(event) > 80 else event)


@app.command()
def profile(
    source: str = typer.Argument(..., help="Path to data file"),
    save: bool = typer.Option(False, "--save", "-s"),
) -> None:
    """Profile a dataset and print its quality report."""
    from ..config import get_settings
    from ..ingestion import DataLoader, DataProfiler, QualityScorer

    cfg = get_settings()
    loader = DataLoader(database_url=cfg.database_url)
    profiler = DataProfiler()
    scorer = QualityScorer()

    df, label = loader.load(source)
    schema = scorer.score(profiler.profile(df, label))
    report = scorer.report(schema)

    console.print(Markdown(report))

    if save:
        out = cfg.workspace_dir / "quality_report.md"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(report, encoding="utf-8")
        console.print(f"[green]Saved: {out}[/green]")


if __name__ == "__main__":
    app()
