"""Tests for the data ingestion and profiling pipeline."""

import io
import json
from pathlib import Path

import polars as pl
import pytest

from src.ingestion.loader import DataLoader
from src.ingestion.profiler import DataProfiler
from src.ingestion.quality import QualityScorer
from src.ingestion.schema import ColumnKind, DataSchema


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_csv(tmp_path: Path) -> Path:
    content = (
        "id,age,salary,department,joined_at\n"
        "1,25,45000,Engineering,2020-01-15\n"
        "2,32,72000,Marketing,2019-06-01\n"
        "3,28,55000,Engineering,2021-03-20\n"
        "4,45,95000,Sales,2018-11-01\n"
        "5,29,,Engineering,2022-07-10\n"  # salary null
        "6,25,45000,Engineering,2020-01-15\n"  # duplicate of row 1
    )
    p = tmp_path / "employees.csv"
    p.write_text(content)
    return p


@pytest.fixture
def sample_df() -> pl.DataFrame:
    return pl.DataFrame({
        "id": [1, 2, 3, 4, 5],
        "age": [25, 32, 28, 45, 29],
        "salary": [45000.0, 72000.0, 55000.0, None, 55000.0],
        "department": ["Eng", "Mkt", "Eng", "Sales", "Eng"],
        "revenue": [100.0, 200.0, 150.0, 800.0, 130.0],
    })


# ---------------------------------------------------------------------------
# DataLoader tests
# ---------------------------------------------------------------------------

class TestDataLoader:
    def test_load_csv(self, sample_csv: Path) -> None:
        loader = DataLoader()
        df, label = loader.load(str(sample_csv))
        assert isinstance(df, pl.DataFrame)
        assert len(df) == 6
        assert "salary" in df.columns
        assert str(sample_csv) in label

    def test_load_parquet(self, tmp_path: Path) -> None:
        original = pl.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
        pq_path = tmp_path / "test.parquet"
        original.write_parquet(pq_path)
        loader = DataLoader()
        df, _ = loader.load(str(pq_path))
        assert df.shape == (3, 2)

    def test_unsupported_suffix_raises(self, tmp_path: Path) -> None:
        p = tmp_path / "data.xml"
        p.write_text("<root/>")
        loader = DataLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load(str(p))

    def test_sql_without_db_raises(self) -> None:
        loader = DataLoader(database_url=None)
        with pytest.raises(RuntimeError, match="DATABASE_URL"):
            loader.load("sql:SELECT 1")


# ---------------------------------------------------------------------------
# DataProfiler tests
# ---------------------------------------------------------------------------

class TestDataProfiler:
    def test_profile_returns_schema(self, sample_df: pl.DataFrame) -> None:
        profiler = DataProfiler()
        schema = profiler.profile(sample_df, source="test")
        assert isinstance(schema, DataSchema)
        assert schema.row_count == 5
        assert schema.col_count == 5

    def test_numeric_column_has_stats(self, sample_df: pl.DataFrame) -> None:
        profiler = DataProfiler()
        schema = profiler.profile(sample_df)
        age_col = next(c for c in schema.columns if c.name == "age")
        assert age_col.kind == ColumnKind.NUMERIC
        assert age_col.mean is not None
        assert age_col.std is not None
        assert age_col.min == 25.0
        assert age_col.max == 45.0

    def test_null_detection(self, sample_df: pl.DataFrame) -> None:
        profiler = DataProfiler()
        schema = profiler.profile(sample_df)
        salary_col = next(c for c in schema.columns if c.name == "salary")
        assert salary_col.null_pct == pytest.approx(0.2)  # 1/5

    def test_categorical_detection(self, sample_df: pl.DataFrame) -> None:
        profiler = DataProfiler()
        schema = profiler.profile(sample_df)
        dept_col = next(c for c in schema.columns if c.name == "department")
        assert dept_col.kind == ColumnKind.CATEGORICAL

    def test_duplicate_detection(self, sample_csv: Path) -> None:
        loader = DataLoader()
        df, label = loader.load(str(sample_csv))
        profiler = DataProfiler()
        schema = profiler.profile(df, source=label)
        assert schema.duplicate_row_pct > 0

    def test_outlier_detection(self) -> None:
        import numpy as np
        values = list(range(100)) + [10000]  # one extreme outlier
        df = pl.DataFrame({"val": values})
        profiler = DataProfiler()
        schema = profiler.profile(df)
        val_col = next(c for c in schema.columns if c.name == "val")
        assert val_col.outlier_pct > 0


# ---------------------------------------------------------------------------
# QualityScorer tests
# ---------------------------------------------------------------------------

class TestQualityScorer:
    def test_perfect_data_high_score(self) -> None:
        df = pl.DataFrame({
            "a": list(range(100)),
            "b": [float(i) for i in range(100)],
        })
        profiler = DataProfiler()
        scorer = QualityScorer()
        schema = scorer.score(profiler.profile(df))
        assert schema.quality_score >= 70

    def test_high_null_lowers_score(self) -> None:
        import polars as pl
        nulls = [None] * 80 + list(range(20))
        df = pl.DataFrame({"a": nulls, "b": list(range(100))})
        profiler = DataProfiler()
        scorer = QualityScorer()
        schema = scorer.score(profiler.profile(df))
        assert schema.quality_score < 60

    def test_report_is_markdown(self, sample_df: pl.DataFrame) -> None:
        profiler = DataProfiler()
        scorer = QualityScorer()
        schema = scorer.score(profiler.profile(sample_df))
        report = scorer.report(schema)
        assert "# Data Quality Report" in report
        assert "| Column |" in report

    def test_schema_summary_for_llm(self, sample_df: pl.DataFrame) -> None:
        profiler = DataProfiler()
        scorer = QualityScorer()
        schema = scorer.score(profiler.profile(sample_df))
        summary = schema.summary_for_llm()
        assert "Shape:" in summary
        assert "Quality score:" in summary
