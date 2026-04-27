"""Quality scoring — spec formula: start at 100, apply penalties, floor at 0."""

from __future__ import annotations

from .schema import ColumnKind, DataSchema

# Quality gate threshold — analysis is HALTED below this
QUALITY_GATE = 40


class QualityScorer:
    """
    Scores dataset quality 0–100.

    Penalty formula (spec):
      - Missingness * 2  : avg null rate (%) × 2 pts deducted
      - Outlier flag * 5 : 5 pts per numeric column with outlier_pct > 10%
      - Duplicate * 10   : 10 pts if duplicate_row_pct > 1%
      - High cardinality * 3 : 3 pts per string column with unique_pct > 90%
    """

    def score(self, schema: DataSchema) -> DataSchema:
        score = 100.0

        # Missingness penalty
        if schema.columns:
            avg_null_pct = (
                sum(c.null_pct for c in schema.columns) / len(schema.columns)
            ) * 100  # convert to 0–100 scale
            score -= avg_null_pct * 2

        # Outlier penalty (5 pts per column with > 10% outliers)
        outlier_cols = [
            c for c in schema.columns
            if c.kind == ColumnKind.NUMERIC and c.outlier_pct > 0.10
        ]
        score -= len(outlier_cols) * 5

        # Duplicate penalty
        if schema.duplicate_row_pct > 0.01:
            score -= 10

        # High-cardinality string column penalty
        high_card_cols = [
            c for c in schema.columns
            if c.kind == ColumnKind.CATEGORICAL and c.unique_pct > 0.90
        ]
        score -= len(high_card_cols) * 3

        schema.quality_score = round(max(0.0, score), 1)
        return schema

    def report(self, schema: DataSchema) -> str:
        gate_status = (
            "FAILED (< 40) — analysis halted"
            if schema.quality_score < QUALITY_GATE
            else "PASSED"
        )
        lines = [
            "# Data Quality Report",
            "",
            f"**Source:** {schema.source}",
            f"**Shape:** {schema.row_count:,} rows x {schema.col_count} columns",
            f"**Overall quality score:** {schema.quality_score:.1f} / 100  [{gate_status}]",
            "",
        ]

        lines += ["## Completeness"]
        high_null = [c for c in schema.columns if c.null_pct > 0.05]
        if high_null:
            for c in sorted(high_null, key=lambda x: -x.null_pct):
                lines.append(f"- `{c.name}`: {c.null_pct*100:.1f}% null")
        else:
            lines.append("- All columns >= 95% complete.")

        lines += ["", "## Uniqueness"]
        lines.append(f"- Duplicate rows: {schema.duplicate_row_pct*100:.2f}%")

        lines += ["", "## Outliers (IQR method, threshold > 10%)"]
        numeric_cols = [c for c in schema.columns if c.kind == ColumnKind.NUMERIC]
        flagged = [c for c in numeric_cols if c.outlier_pct > 0.10]
        if flagged:
            for c in sorted(flagged, key=lambda x: -x.outlier_pct):
                lines.append(f"- `{c.name}`: {c.outlier_pct*100:.1f}% outliers")
        else:
            lines.append("- No numeric columns with >10% outliers.")

        if schema.ingest_warnings:
            lines += ["", "## Warnings"]
            for w in schema.ingest_warnings:
                lines.append(f"- {w}")

        lines += ["", "## Column Summary", ""]
        lines.append("| Column | Kind | Null% | Unique% | Mean | Std |")
        lines.append("|--------|------|-------|---------|------|-----|")
        for c in schema.columns:
            mean_s = f"{c.mean:.3f}" if c.mean is not None else "-"
            std_s = f"{c.std:.3f}" if c.std is not None else "-"
            lines.append(
                f"| `{c.name}` | {c.kind.value} | {c.null_pct*100:.1f}% "
                f"| {c.unique_pct*100:.1f}% | {mean_s} | {std_s} |"
            )

        return "\n".join(lines)
