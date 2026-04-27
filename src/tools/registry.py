"""Tool registry — OpenAI tool definitions + execution handlers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..logging_config import get_logger
from .sandbox import ExecutionResult, SandboxExecutor

logger = get_logger(__name__)


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[..., Any]

    def to_openai_spec(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def list_specs(self) -> list[dict[str, Any]]:
        return [t.to_openai_spec() for t in self._tools.values()]

    def call(self, name: str, **kwargs: Any) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ValueError(f"Unknown tool: '{name}'. Available: {list(self._tools)}")

        # Defensive: parse string-encoded JSON arrays/objects from sloppy LLMs
        parsed_kwargs: dict[str, Any] = {}
        for k, v in kwargs.items():
            if isinstance(v, str) and (v.startswith("[") or v.startswith("{")):
                try:
                    parsed_kwargs[k] = json.loads(v)
                except json.JSONDecodeError:
                    parsed_kwargs[k] = v
            else:
                parsed_kwargs[k] = v

        logger.info("tool_called", name=name)
        return tool.handler(**parsed_kwargs)


# ---------------------------------------------------------------------------
# Tool factories
# ---------------------------------------------------------------------------

def _make_python_executor(executor: SandboxExecutor) -> Tool:
    def handler(code: str) -> dict[str, Any]:
        result: ExecutionResult = executor.execute(code)
        return result.to_dict()

    return Tool(
        name="python_executor",
        description=(
            "Execute Python code in a sandboxed environment. "
            "Available libraries: polars, pandas, numpy, scipy, statsmodels, scikit-learn, "
            "plotly, matplotlib, seaborn. "
            "Two path variables are pre-injected: WORKSPACE (session dir) and CHARTS_DIR. "
            "Save all charts to CHARTS_DIR as .png files using matplotlib/seaborn, "
            "or as .html using plotly. "
            "Print all numerical findings to stdout — they are returned to you."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python 3.10 code to execute."}
            },
            "required": ["code"],
        },
        handler=handler,
    )


def _make_correlation_matrix() -> Tool:
    def handler(
        file_path: str,
        columns: list[str] | None = None,
        method: str = "pearson",
    ) -> dict[str, Any]:
        import polars as pl
        import pandas as pd
        from scipy import stats

        # Read CSV and select requested columns
        df_pl = pl.read_csv(file_path)
        if columns:
            df_pl = df_pl.select(columns)

        # Convert to pandas and auto-select only numeric columns
        df = df_pl.to_pandas().select_dtypes(include="number")

        if df.empty:
            return {
                "method": method,
                "n": 0,
                "matrix": {},
                "p_values": {},
                "strong_pairs": [],
                "warning": "No numeric columns found after filtering.",
            }

        corr = df.corr(method=method)

        # p-values via scipy
        cols = list(df.columns)
        pvals: dict[str, dict[str, float]] = {}
        for c1 in cols:
            pvals[c1] = {}
            for c2 in cols:
                if c1 == c2:
                    pvals[c1][c2] = 0.0
                else:
                    _, p = stats.pearsonr(df[c1].dropna(), df[c2].dropna())
                    pvals[c1][c2] = round(float(p), 6)

        strong = [
            {"col_a": c1, "col_b": c2, "r": round(corr.loc[c1, c2], 4),
             "p_value": pvals[c1][c2]}
            for c1 in cols for c2 in cols
            if c1 < c2 and abs(corr.loc[c1, c2]) >= 0.4
        ]
        return {
            "method": method,
            "n": len(df),
            "numeric_columns": cols,
            "matrix": corr.round(4).to_dict(),
            "p_values": pvals,
            "strong_pairs": sorted(strong, key=lambda x: -abs(x["r"])),
        }

    return Tool(
        name="correlation_matrix",
        description=(
            "Read a CSV file and compute Pearson or Spearman correlations across numeric columns. "
            "Automatically filters to numeric columns only (excludes categorical data). "
            "Pass file_path pointing to the data CSV, optional columns list to restrict analysis. "
            "Returns full correlation matrix, p-values, and highlights pairs with |r| >= 0.4."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the CSV file.",
                },
                "columns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional list of column names to include. Defaults to all columns (auto-filters to numeric).",
                },
                "method": {
                    "type": "string",
                    "enum": ["pearson", "spearman"],
                    "default": "pearson",
                    "description": "Correlation method to use.",
                },
            },
            "required": ["file_path"],
        },
        handler=handler,
    )


def _make_regression_analysis() -> Tool:
    def handler(
        file_path: str,
        target: str,
        features: list[str],
        model_type: str = "ols",
    ) -> dict[str, Any]:
        import polars as pl
        import pandas as pd
        import numpy as np
        import statsmodels.api as sm

        # Read CSV and select relevant columns
        df_pl = pl.read_csv(file_path).select([target] + features).drop_nulls()
        df = df_pl.to_pandas()

        # Coerce to numeric and drop any remaining NaN
        y_series = pd.to_numeric(df[target], errors="coerce")
        X_df = df[features].apply(pd.to_numeric, errors="coerce")
        valid = y_series.notna() & X_df.notna().all(axis=1)

        y_arr = y_series[valid].to_numpy(dtype=float)
        X_df = X_df[valid]

        if len(y_arr) < 2 or X_df.empty:
            raise ValueError(
                f"Insufficient valid numeric data after filtering. "
                f"n={len(y_arr)}, features={len(X_df.columns)}"
            )

        df_with_const = sm.add_constant(X_df)

        if model_type == "ols":
            fitted = sm.OLS(y_arr, df_with_const).fit()
        elif model_type == "logistic":
            fitted = sm.Logit(y_arr, df_with_const).fit(disp=False)
        else:
            raise ValueError(f"Unknown model_type: {model_type}. Use 'ols' or 'logistic'.")

        ci = fitted.conf_int()
        coefs = {
            name: {
                "coef": round(float(fitted.params[name]), 6),
                "p_value": round(float(fitted.pvalues[name]), 6),
                "ci_low": round(float(ci.loc[name, 0]), 6),
                "ci_high": round(float(ci.loc[name, 1]), 6),
                "significant": bool(fitted.pvalues[name] < 0.05),
            }
            for name in fitted.params.index
        }

        result: dict[str, Any] = {
            "model_type": model_type,
            "target": target,
            "features": features,
            "n": int(fitted.nobs),
            "coefficients": coefs,
        }
        if model_type == "ols":
            result["r_squared"] = round(float(fitted.rsquared), 4)
            result["adj_r_squared"] = round(float(fitted.rsquared_adj), 4)
            result["f_statistic"] = round(float(fitted.fvalue), 4)
            result["f_p_value"] = round(float(fitted.f_pvalue), 6)
            result["aic"] = round(float(fitted.aic), 2)

        return result

    return Tool(
        name="regression_analysis",
        description=(
            "Read a CSV file and fit OLS or logistic regression to predict a target from features. "
            "Pass file_path pointing to the data CSV, target column name, list of feature column names, "
            "and model_type ('ols' for continuous targets, 'logistic' for binary classification). "
            "Returns coefficients, p-values, 95% confidence intervals, R-squared, and sample size. "
            "Automatically coerces columns to numeric and handles missing values."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "file_path": {
                    "type": "string",
                    "description": "Absolute path to the CSV file.",
                },
                "target": {
                    "type": "string",
                    "description": "Name of the target column to predict.",
                },
                "features": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of feature column names to use as predictors.",
                },
                "model_type": {
                    "type": "string",
                    "enum": ["ols", "logistic"],
                    "default": "ols",
                    "description": "'ols' for continuous targets, 'logistic' for binary targets.",
                },
            },
            "required": ["file_path", "target", "features"],
        },
        handler=handler,
    )


def _make_data_transform() -> Tool:
    def handler(
        columns_data: dict[str, list[Any]],
        operations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        import numpy as np
        import pandas as pd
        from sklearn.preprocessing import StandardScaler, OneHotEncoder
        from sklearn.impute import SimpleImputer

        df = pd.DataFrame(columns_data)
        results: dict[str, Any] = {"operations_applied": []}

        for op in operations:
            op_type = op.get("type")
            cols = op.get("columns", list(df.columns))

            if op_type == "standardize":
                scaler = StandardScaler()
                df[cols] = scaler.fit_transform(df[cols])
                results["operations_applied"].append(
                    {"type": "standardize", "columns": cols,
                     "means": dict(zip(cols, scaler.mean_.tolist())),
                     "stds": dict(zip(cols, scaler.scale_.tolist()))}
                )

            elif op_type == "impute":
                strategy = op.get("strategy", "mean")
                imputer = SimpleImputer(strategy=strategy)
                df[cols] = imputer.fit_transform(df[cols])
                results["operations_applied"].append(
                    {"type": "impute", "strategy": strategy, "columns": cols}
                )

            elif op_type == "one_hot_encode":
                enc = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
                encoded = enc.fit_transform(df[cols])
                new_cols = enc.get_feature_names_out(cols).tolist()
                encoded_df = pd.DataFrame(encoded, columns=new_cols, index=df.index)
                df = pd.concat([df.drop(columns=cols), encoded_df], axis=1)
                results["operations_applied"].append(
                    {"type": "one_hot_encode", "original_columns": cols,
                     "new_columns": new_cols}
                )

            elif op_type == "log_transform":
                for col in cols:
                    df[col] = np.log1p(df[col].clip(lower=0))
                results["operations_applied"].append(
                    {"type": "log_transform", "columns": cols}
                )
            else:
                raise ValueError(
                    f"Unknown operation: {op_type}. "
                    "Supported: standardize, impute, one_hot_encode, log_transform"
                )

        results["transformed_data"] = df.to_dict(orient="list")
        results["shape"] = list(df.shape)
        results["columns"] = list(df.columns)
        return results

    return Tool(
        name="data_transform",
        description=(
            "Transform data columns: standardize (StandardScaler), impute missing values "
            "(mean/median/most_frequent), one_hot_encode categorical columns, or log_transform. "
            "Pass a list of operations to apply in sequence. "
            "Returns the transformed data as a dict."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "columns_data": {
                    "type": "object",
                    "description": "Dict of column_name -> list of values.",
                },
                "operations": {
                    "type": "array",
                    "description": "Ordered list of transformation ops.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "type": {
                                "type": "string",
                                "enum": ["standardize", "impute", "one_hot_encode", "log_transform"],
                            },
                            "columns": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Columns to apply this operation to.",
                            },
                            "strategy": {
                                "type": "string",
                                "description": "For impute: mean | median | most_frequent",
                            },
                        },
                        "required": ["type"],
                    },
                },
            },
            "required": ["columns_data", "operations"],
        },
        handler=handler,
    )


def build_default_registry(
    session_dir: Path | None = None,
    data_path: Path | None = None,
) -> ToolRegistry:
    registry = ToolRegistry()
    executor = SandboxExecutor(session_dir=session_dir, data_path=data_path)

    registry.register(_make_python_executor(executor))
    registry.register(_make_correlation_matrix())
    registry.register(_make_regression_analysis())
    registry.register(_make_data_transform())

    logger.info("tool_registry_built", tools=list(registry._tools.keys()))
    return registry
