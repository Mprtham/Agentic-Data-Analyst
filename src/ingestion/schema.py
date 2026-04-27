"""Data schema and metadata models."""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ColumnKind(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    UNKNOWN = "unknown"


@dataclass
class ColumnMeta:
    name: str
    kind: ColumnKind
    dtype: str
    null_pct: float          # 0.0–1.0
    unique_count: int
    unique_pct: float
    sample_values: list[Any] = field(default_factory=list)
    min: Any = None
    max: Any = None
    mean: float | None = None
    std: float | None = None
    outlier_pct: float = 0.0  # fraction of rows flagged as outliers (IQR)


@dataclass
class DataSchema:
    source: str              # file path, URL, or "sql:<query>"
    row_count: int
    col_count: int
    columns: list[ColumnMeta] = field(default_factory=list)
    duplicate_row_pct: float = 0.0
    memory_mb: float = 0.0
    ingest_warnings: list[str] = field(default_factory=list)

    # Derived quality score 0–100
    quality_score: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "row_count": self.row_count,
            "col_count": self.col_count,
            "duplicate_row_pct": round(self.duplicate_row_pct, 4),
            "memory_mb": round(self.memory_mb, 2),
            "quality_score": round(self.quality_score, 1),
            "ingest_warnings": self.ingest_warnings,
            "columns": [
                {
                    "name": c.name,
                    "kind": c.kind.value,
                    "dtype": c.dtype,
                    "null_pct": round(c.null_pct, 4),
                    "unique_count": c.unique_count,
                    "unique_pct": round(c.unique_pct, 4),
                    "min": c.min,
                    "max": c.max,
                    "mean": round(c.mean, 4) if c.mean is not None else None,
                    "std": round(c.std, 4) if c.std is not None else None,
                    "outlier_pct": round(c.outlier_pct, 4),
                    "sample_values": c.sample_values[:5],
                }
                for c in self.columns
            ],
        }

    def summary_for_llm(self) -> str:
        """Compact text representation suitable for injection into an LLM prompt."""
        lines = [
            f"Dataset: {self.source}",
            f"Shape: {self.row_count:,} rows × {self.col_count} columns",
            f"Quality score: {self.quality_score:.1f}/100",
            f"Duplicate rows: {self.duplicate_row_pct*100:.1f}%",
            f"Memory: {self.memory_mb:.1f} MB",
            "",
            "Columns:",
        ]
        for c in self.columns:
            null_flag = f" [!] {c.null_pct*100:.0f}% null" if c.null_pct > 0.05 else ""
            out_flag = f" [!] {c.outlier_pct*100:.0f}% outliers" if c.outlier_pct > 0.05 else ""
            stats = ""
            if c.mean is not None:
                stats = f" | mean={c.mean:.2f} std={c.std:.2f}"
            lines.append(
                f"  {c.name} ({c.kind.value}, {c.dtype}){null_flag}{out_flag}{stats}"
            )
        if self.ingest_warnings:
            lines += ["", "Warnings:"] + [f"  ! {w}" for w in self.ingest_warnings]
        return "\n".join(lines)
