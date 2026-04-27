"""Tests for session persistence and state management."""

import json
from pathlib import Path

import pytest

from src.engine.session import AnalysisSession


class TestAnalysisSession:
    def test_session_creation(self) -> None:
        s = AnalysisSession(business_question="Test?")
        assert s.business_question == "Test?"
        assert s.status == "active"
        assert len(s.session_id) == 8

    def test_add_finding(self) -> None:
        s = AnalysisSession()
        s.add_finding("Revenue correlates with season (r=0.82)")
        assert len(s.findings) == 1

    def test_save_and_load(self, tmp_path: Path) -> None:
        s = AnalysisSession(
            business_question="What drives churn?",
            dataset_source="customers.csv",
            quality_score=78.5,
        )
        s.add_finding("Tenure < 3 months is the strongest churn predictor")
        s.add_artifact("/workspace/outputs/chart.html")

        saved = s.save(tmp_path)
        assert saved.exists()

        loaded = AnalysisSession.load(saved)
        assert loaded.session_id == s.session_id
        assert loaded.business_question == s.business_question
        assert loaded.quality_score == 78.5
        assert len(loaded.findings) == 1
        assert len(loaded.artifacts) == 1

    def test_load_latest(self, tmp_path: Path) -> None:
        for i in range(3):
            s = AnalysisSession(business_question=f"Question {i}")
            s.save(tmp_path)

        latest = AnalysisSession.load_latest(tmp_path)
        assert latest is not None
        assert "Question" in latest.business_question

    def test_load_latest_empty_dir(self, tmp_path: Path) -> None:
        result = AnalysisSession.load_latest(tmp_path)
        assert result is None

    def test_no_duplicate_artifacts(self) -> None:
        s = AnalysisSession()
        s.add_artifact("chart.html")
        s.add_artifact("chart.html")
        assert len(s.artifacts) == 1
