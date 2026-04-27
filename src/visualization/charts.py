"""
Programmatic chart builder using Plotly (interactive) and Seaborn (static).

Used by the reporting layer to auto-generate charts from analysis results.
The agent also generates charts directly via python_executor;
this module is for the reporting pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from ..logging_config import get_logger

logger = get_logger(__name__)


class ChartBuilder:
    def __init__(self, output_dir: Path) -> None:
        self._out = output_dir
        self._out.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Distribution charts
    # ------------------------------------------------------------------

    def histogram(
        self,
        values: list[float],
        column_name: str,
        nbins: int = 40,
        filename: str | None = None,
    ) -> Path:
        fig = px.histogram(
            x=values,
            nbins=nbins,
            title=f"Distribution of {column_name}",
            labels={"x": column_name, "y": "Count"},
            template="plotly_white",
        )
        fig.update_layout(bargap=0.05)
        return self._save(fig, filename or f"hist_{column_name}.html")

    def box_plot(
        self,
        data: dict[str, list[float]],
        title: str = "Box Plot",
        filename: str | None = None,
    ) -> Path:
        fig = go.Figure()
        for col, vals in data.items():
            fig.add_trace(go.Box(y=vals, name=col, boxpoints="outliers"))
        fig.update_layout(title=title, template="plotly_white", yaxis_title="Value")
        return self._save(fig, filename or "boxplot.html")

    # ------------------------------------------------------------------
    # Relationship charts
    # ------------------------------------------------------------------

    def scatter(
        self,
        x: list[float],
        y: list[float],
        x_label: str,
        y_label: str,
        color: list[Any] | None = None,
        trendline: bool = True,
        filename: str | None = None,
    ) -> Path:
        fig = px.scatter(
            x=x, y=y, color=color,
            labels={"x": x_label, "y": y_label},
            title=f"{y_label} vs {x_label}",
            template="plotly_white",
            trendline="ols" if trendline else None,
        )
        return self._save(fig, filename or f"scatter_{x_label}_vs_{y_label}.html")

    def correlation_heatmap(
        self,
        matrix: dict[str, dict[str, float]],
        filename: str | None = None,
    ) -> Path:
        import pandas as pd
        df = pd.DataFrame(matrix)
        fig = px.imshow(
            df,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            title="Correlation Matrix",
            template="plotly_white",
            text_auto=".2f",
        )
        return self._save(fig, filename or "correlation_heatmap.html")

    # ------------------------------------------------------------------
    # Time series
    # ------------------------------------------------------------------

    def time_series(
        self,
        x: list[Any],
        y: list[float],
        title: str = "Time Series",
        trend: list[float] | None = None,
        filename: str | None = None,
    ) -> Path:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="Observed"))
        if trend:
            fig.add_trace(go.Scatter(x=x, y=trend, mode="lines",
                                      name="Trend", line=dict(dash="dash")))
        fig.update_layout(title=title, template="plotly_white")
        return self._save(fig, filename or "timeseries.html")

    def decomposition_plot(
        self,
        observed: list[float],
        trend: list[float],
        seasonal: list[float],
        residual: list[float],
        filename: str | None = None,
    ) -> Path:
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                             subplot_titles=["Observed", "Trend", "Seasonal", "Residual"])
        for i, (name, data) in enumerate([
            ("Observed", observed), ("Trend", trend),
            ("Seasonal", seasonal), ("Residual", residual),
        ], 1):
            fig.add_trace(go.Scatter(y=data, mode="lines", name=name), row=i, col=1)
        fig.update_layout(height=800, title="Time Series Decomposition",
                          template="plotly_white", showlegend=False)
        return self._save(fig, filename or "decomposition.html")

    # ------------------------------------------------------------------
    # Categorical / comparison
    # ------------------------------------------------------------------

    def bar_chart(
        self,
        categories: list[str],
        values: list[float],
        title: str = "Bar Chart",
        x_label: str = "Category",
        y_label: str = "Value",
        filename: str | None = None,
    ) -> Path:
        fig = px.bar(
            x=categories, y=values,
            labels={"x": x_label, "y": y_label},
            title=title, template="plotly_white",
        )
        return self._save(fig, filename or "barchart.html")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _save(self, fig: go.Figure, filename: str) -> Path:
        path = self._out / filename
        fig.write_html(str(path), include_plotlyjs="cdn")
        logger.info("chart_saved", path=str(path))
        return path
