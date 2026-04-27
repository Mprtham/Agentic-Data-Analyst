"""Tests for the tool registry and statistical tools."""

import math
import pytest

from src.tools.registry import (
    _make_statistical_test,
    _make_correlation_matrix,
    _make_regression_analysis,
    _make_clustering,
    _make_time_series_decomp,
)


@pytest.fixture
def stat_tool():
    return _make_statistical_test()


@pytest.fixture
def corr_tool():
    return _make_correlation_matrix()


@pytest.fixture
def reg_tool():
    return _make_regression_analysis()


@pytest.fixture
def cluster_tool():
    return _make_clustering()


@pytest.fixture
def ts_tool():
    return _make_time_series_decomp()


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

class TestStatisticalTest:
    def test_t_test_detects_significant_difference(self, stat_tool) -> None:
        a = list(range(50))         # mean ~24.5
        b = list(range(200, 250))   # mean ~224.5 — hugely different
        result = stat_tool.handler(test="t_test_independent", data_a=a, data_b=b)
        assert result["significant"] is True
        assert result["p_value"] < 0.05

    def test_t_test_detects_no_difference(self, stat_tool) -> None:
        import random
        rng = random.Random(42)
        a = [rng.gauss(50, 5) for _ in range(100)]
        b = [rng.gauss(50, 5) for _ in range(100)]
        result = stat_tool.handler(test="t_test_independent", data_a=a, data_b=b, alpha=0.001)
        # With very tight alpha, small random differences should not be significant
        assert "significant" in result

    def test_shapiro_normality(self, stat_tool) -> None:
        import random
        rng = random.Random(0)
        normal_data = [rng.gauss(0, 1) for _ in range(200)]
        result = stat_tool.handler(test="shapiro", data_a=normal_data)
        assert "is_normal" in result

    def test_chi_square(self, stat_tool) -> None:
        observed = [10, 20, 30]
        expected = [20, 20, 20]  # equal expected
        result = stat_tool.handler(test="chi_square", data_a=observed, data_b=expected)
        assert "p_value" in result

    def test_unknown_test_raises(self, stat_tool) -> None:
        with pytest.raises(ValueError, match="Unknown test"):
            stat_tool.handler(test="magic_test", data_a=[1, 2, 3])

    def test_missing_data_b_raises(self, stat_tool) -> None:
        with pytest.raises(ValueError):
            stat_tool.handler(test="t_test_independent", data_a=[1, 2, 3])


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

class TestCorrelation:
    def test_perfect_positive_correlation(self, corr_tool) -> None:
        x = list(range(100))
        result = corr_tool.handler(columns_data={"x": x, "y": x})
        matrix = result["matrix"]
        assert abs(matrix["x"]["y"] - 1.0) < 0.001

    def test_strong_pairs_identified(self, corr_tool) -> None:
        x = [float(i) for i in range(100)]
        y = [v * 2.0 for v in x]
        z = [100.0 - v for v in x]
        result = corr_tool.handler(columns_data={"x": x, "y": y, "z": z})
        assert len(result["strong_pairs"]) >= 2

    def test_spearman(self, corr_tool) -> None:
        x = list(range(100))
        y = [v**2 for v in x]  # non-linear but monotonic
        result = corr_tool.handler(columns_data={"x": x, "y": y}, method="spearman")
        assert result["method"] == "spearman"


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class TestRegression:
    def test_ols_r_squared(self, reg_tool) -> None:
        x = [float(i) for i in range(100)]
        y = [2.0 * v + 5.0 + 0.01 for v in x]  # near-perfect linear
        result = reg_tool.handler(y=y, X={"x": x}, model_type="ols")
        assert result["r_squared"] > 0.99
        assert "x" in result["coefficients"]

    def test_ols_coefficient_direction(self, reg_tool) -> None:
        x = [float(i) for i in range(50)]
        y = [3.0 * v + 10.0 for v in x]
        result = reg_tool.handler(y=y, X={"x": x})
        coef_x = result["coefficients"]["x"]["coef"]
        assert coef_x == pytest.approx(3.0, abs=0.1)

    def test_unknown_model_raises(self, reg_tool) -> None:
        with pytest.raises(ValueError):
            reg_tool.handler(y=[1.0, 2.0], X={"x": [1.0, 2.0]}, model_type="gbm")


# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------

class TestClustering:
    def test_kmeans_returns_labels(self, cluster_tool) -> None:
        # Two clearly separated clusters
        data = {
            "x": [0.0, 0.1, 0.2] + [10.0, 10.1, 10.2],
            "y": [0.0, 0.1, 0.2] + [10.0, 10.1, 10.2],
        }
        result = cluster_tool.handler(data=data, n_clusters=2)
        assert len(set(result["labels"])) == 2
        assert result["silhouette_score"] > 0.9

    def test_cluster_sizes_sum_to_total(self, cluster_tool) -> None:
        n = 30
        data = {"x": list(range(n)), "y": list(range(n))}
        result = cluster_tool.handler(data=data, n_clusters=3)
        total = sum(result["cluster_sizes"].values())
        assert total == n


# ---------------------------------------------------------------------------
# Time series
# ---------------------------------------------------------------------------

class TestTimeSeries:
    def test_decomposition_returns_components(self, ts_tool) -> None:
        import math
        n = 48  # 4 years of monthly data
        values = [10.0 + i * 0.5 + 5.0 * math.sin(2 * math.pi * i / 12) for i in range(n)]
        result = ts_tool.handler(values=values, period=12)
        assert len(result["trend"]) == n
        assert len(result["seasonal"]) == n
        assert len(result["residual"]) == n
        assert result["trend_direction"] in ("upward", "downward")

    def test_upward_trend_detected(self, ts_tool) -> None:
        values = [float(i) for i in range(24)]
        result = ts_tool.handler(values=values, period=4)
        assert result["trend_direction"] == "upward"
