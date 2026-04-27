"""Automatic data profiling — produces DataSchema from a Polars DataFrame."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import polars as pl

from ..logging_config import get_logger
from .schema import ColumnKind, ColumnMeta, DataSchema

logger = get_logger(__name__)

# A column is "high cardinality" if its unique ratio exceeds this threshold
_HIGH_CARD_THRESHOLD = 0.95
# A column is "categorical" if its unique ratio is below this threshold
_CAT_THRESHOLD = 0.10
# IQR multiplier for outlier detection
_IQR_FACTOR = 1.5


class DataProfiler:
    """Profiles a Polars DataFrame and returns a DataSchema."""

    def profile(self, df: pl.DataFrame, source: str = "unknown") -> DataSchema:
        logger.info("profiling_dataset", source=source, rows=len(df), cols=len(df.columns))

        cols: list[ColumnMeta] = []
        warnings: list[str] = []

        for col_name in df.columns:
            series = df[col_name]
            meta = self._profile_column(series, len(df))
            cols.append(meta)

            if meta.null_pct > 0.5:
                warnings.append(f"Column '{col_name}' is {meta.null_pct*100:.0f}% null")
            if meta.unique_pct > _HIGH_CARD_THRESHOLD and meta.kind == ColumnKind.CATEGORICAL:
                warnings.append(f"Column '{col_name}' may be an ID column (high cardinality)")

        dup_count = len(df) - len(df.unique())
        dup_pct = dup_count / len(df) if len(df) > 0 else 0.0
        if dup_pct > 0.01:
            warnings.append(f"{dup_pct*100:.1f}% duplicate rows detected")

        memory_mb = df.estimated_size("mb")

        schema = DataSchema(
            source=source,
            row_count=len(df),
            col_count=len(df.columns),
            columns=cols,
            duplicate_row_pct=dup_pct,
            memory_mb=memory_mb,
            ingest_warnings=warnings,
        )
        return schema

    # ------------------------------------------------------------------
    # Column-level profiling
    # ------------------------------------------------------------------

    def _profile_column(self, series: pl.Series, total_rows: int) -> ColumnMeta:
        null_count = series.null_count()
        null_pct = null_count / total_rows if total_rows > 0 else 0.0
        non_null = series.drop_nulls()
        unique_count = series.n_unique()
        unique_pct = unique_count / total_rows if total_rows > 0 else 0.0

        kind = self._infer_kind(series)
        sample = [v for v in series.drop_nulls().head(5).to_list()]

        min_val: Any = None
        max_val: Any = None
        mean_val: float | None = None
        std_val: float | None = None
        outlier_pct: float = 0.0

        if kind == ColumnKind.NUMERIC and len(non_null) > 0:
            arr = non_null.cast(pl.Float64).to_numpy()
            min_val = float(arr.min())
            max_val = float(arr.max())
            mean_val = float(arr.mean())
            std_val = float(arr.std()) if len(arr) > 1 else 0.0
            outlier_pct = self._iqr_outlier_pct(arr)
        elif kind in {ColumnKind.DATETIME} and len(non_null) > 0:
            min_val = str(non_null.min())
            max_val = str(non_null.max())
        elif kind == ColumnKind.CATEGORICAL and len(non_null) > 0:
            min_val = None
            max_val = None

        return ColumnMeta(
            name=series.name,
            kind=kind,
            dtype=str(series.dtype),
            null_pct=null_pct,
            unique_count=unique_count,
            unique_pct=unique_pct,
            sample_values=sample,
            min=min_val,
            max=max_val,
            mean=mean_val,
            std=std_val,
            outlier_pct=outlier_pct,
        )

    def _infer_kind(self, series: pl.Series) -> ColumnKind:
        dtype = series.dtype
        if dtype in (
            pl.Int8, pl.Int16, pl.Int32, pl.Int64,
            pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
            pl.Float32, pl.Float64,
        ):
            return ColumnKind.NUMERIC
        if dtype == pl.Boolean:
            return ColumnKind.BOOLEAN
        if dtype in (pl.Date, pl.Datetime, pl.Duration, pl.Time):
            return ColumnKind.DATETIME
        if dtype == pl.Utf8:
            # Heuristic: try parsing as datetime
            sample = series.drop_nulls().head(20)
            try:
                parsed = sample.str.to_datetime(strict=False)
                if parsed.null_count() < len(sample) * 0.5:
                    return ColumnKind.DATETIME
            except Exception:
                pass
            # Heuristic: low cardinality → categorical
            total = len(series)
            unique_pct = series.n_unique() / total if total > 0 else 0
            if unique_pct <= _CAT_THRESHOLD:
                return ColumnKind.CATEGORICAL
            # Long strings → text
            avg_len = sample.str.len_chars().mean() or 0
            if avg_len > 50:
                return ColumnKind.TEXT
            return ColumnKind.CATEGORICAL
        return ColumnKind.UNKNOWN

    @staticmethod
    def _iqr_outlier_pct(arr: np.ndarray) -> float:
        if len(arr) < 4:
            return 0.0
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        if iqr == 0:
            return 0.0
        lower = q1 - _IQR_FACTOR * iqr
        upper = q3 + _IQR_FACTOR * iqr
        outliers = np.sum((arr < lower) | (arr > upper))
        return float(outliers) / len(arr)
