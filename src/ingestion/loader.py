"""Data loading from CSV, Excel, Parquet, SQL, and REST APIs."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx
import polars as pl
import pandas as pd
from sqlalchemy import create_engine, text

from ..logging_config import get_logger

logger = get_logger(__name__)

_SUPPORTED_SUFFIXES = {".csv", ".tsv", ".parquet", ".xlsx", ".xls", ".json", ".jsonl"}


class DataLoader:
    """Loads data from various sources into a Polars DataFrame."""

    def __init__(self, database_url: str | None = None) -> None:
        self._db_url = database_url

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self, source: str, **kwargs: Any) -> tuple[pl.DataFrame, str]:
        """
        Load data from *source* and return (DataFrame, canonical_source_label).

        source can be:
          - a local file path
          - a SQL query prefixed with "sql:" e.g. "sql:SELECT * FROM orders LIMIT 1000"
          - an HTTP/HTTPS URL (returns JSON or CSV)
        """
        if source.startswith("sql:"):
            return self._load_sql(source[4:].strip())
        if source.startswith("http://") or source.startswith("https://"):
            return self._load_url(source, **kwargs)
        return self._load_file(Path(source), **kwargs)

    # ------------------------------------------------------------------
    # File loading
    # ------------------------------------------------------------------

    def _load_file(self, path: Path, **kwargs: Any) -> tuple[pl.DataFrame, str]:
        suffix = path.suffix.lower()
        if suffix not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported file type: {suffix}. Supported: {_SUPPORTED_SUFFIXES}")

        logger.info("loading_file", path=str(path), suffix=suffix)

        if suffix in {".csv", ".tsv"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pl.read_csv(path, separator=sep, infer_schema_length=10_000, **kwargs)
        elif suffix == ".parquet":
            df = pl.read_parquet(path, **kwargs)
        elif suffix in {".xlsx", ".xls"}:
            # Polars uses pandas under the hood for Excel
            pdf = pd.read_excel(path, **kwargs)
            df = pl.from_pandas(pdf)
        elif suffix == ".json":
            df = pl.read_json(path, **kwargs)
        elif suffix == ".jsonl":
            df = pl.read_ndjson(path, **kwargs)
        else:
            raise ValueError(f"Unhandled suffix: {suffix}")

        logger.info("file_loaded", rows=len(df), cols=len(df.columns))
        return df, str(path)

    # ------------------------------------------------------------------
    # SQL loading
    # ------------------------------------------------------------------

    def _load_sql(self, query: str) -> tuple[pl.DataFrame, str]:
        if not self._db_url:
            raise RuntimeError(
                "DATABASE_URL is not configured. Set it in .env to use SQL loading."
            )
        logger.info("loading_sql", query_preview=query[:80])
        engine = create_engine(self._db_url)
        with engine.connect() as conn:
            pdf = pd.read_sql(text(query), conn)
        df = pl.from_pandas(pdf)
        logger.info("sql_loaded", rows=len(df), cols=len(df.columns))
        return df, f"sql:{query[:60]}..."

    # ------------------------------------------------------------------
    # REST API / URL loading
    # ------------------------------------------------------------------

    def _load_url(self, url: str, **kwargs: Any) -> tuple[pl.DataFrame, str]:
        logger.info("loading_url", url=url)
        response = httpx.get(url, follow_redirects=True, timeout=30)
        response.raise_for_status()

        content_type = response.headers.get("content-type", "")
        if "json" in content_type or url.endswith(".json"):
            data = response.json()
            if isinstance(data, list):
                df = pl.DataFrame(data)
            elif isinstance(data, dict):
                # Try to find a list-valued key (e.g. {"results": [...]})
                list_keys = [k for k, v in data.items() if isinstance(v, list)]
                if list_keys:
                    df = pl.DataFrame(data[list_keys[0]])
                else:
                    df = pl.DataFrame([data])
            else:
                raise ValueError(f"Cannot convert JSON of type {type(data)} to DataFrame")
        elif "csv" in content_type or url.endswith(".csv"):
            df = pl.read_csv(io.StringIO(response.text))
        else:
            # Try CSV first, fall back to JSON
            try:
                df = pl.read_csv(io.StringIO(response.text))
            except Exception:
                data = json.loads(response.text)
                df = pl.DataFrame(data if isinstance(data, list) else [data])

        parsed = urlparse(url)
        label = f"{parsed.netloc}{parsed.path}"
        logger.info("url_loaded", rows=len(df), cols=len(df.columns))
        return df, label

    # ------------------------------------------------------------------
    # S3 / cloud storage stub — extend here
    # ------------------------------------------------------------------

    def load_s3(self, s3_uri: str) -> tuple[pl.DataFrame, str]:
        """Load from s3://bucket/key. Requires boto3 and AWS credentials."""
        try:
            import boto3  # type: ignore[import]
            import s3fs  # type: ignore[import]
        except ImportError as exc:
            raise ImportError("Install boto3 and s3fs for S3 support") from exc

        fs = s3fs.S3FileSystem()
        path = s3_uri.replace("s3://", "")
        suffix = Path(path).suffix.lower()
        if suffix == ".parquet":
            with fs.open(path, "rb") as f:
                df = pl.read_parquet(f)
        elif suffix in {".csv", ".tsv"}:
            with fs.open(path, "rb") as f:
                df = pl.read_csv(f)
        else:
            raise ValueError(f"S3 loading not implemented for {suffix}")
        return df, s3_uri
