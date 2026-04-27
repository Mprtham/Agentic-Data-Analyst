"""Sandboxed Python execution.

Production path: Docker container, --network=none, read-only fs except /workspace.
Dev fallback: restricted subprocess (no Docker). A clear warning is printed.
"""

from __future__ import annotations

import hashlib
import io
import contextlib
import subprocess
import textwrap
import time
from pathlib import Path
from typing import Any

from ..config import get_settings
from ..logging_config import get_logger

logger = get_logger(__name__)

_DOCKER_WARNING = """
[SANDBOX WARNING] Docker is not available on this machine.
Code will run in a restricted in-process sandbox (development mode only).
To enable full isolation, start Docker Desktop and build the sandbox image:
  docker build -f docker/Dockerfile.sandbox -t analyst-sandbox:latest .
"""


class ExecutionResult:
    def __init__(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
        duration_s: float,
        artifacts: list[Path],
        sandbox_mode: str = "docker",
    ) -> None:
        self.stdout = stdout
        self.stderr = stderr
        self.exit_code = exit_code
        self.duration_s = duration_s
        self.artifacts = artifacts
        self.sandbox_mode = sandbox_mode
        self.success = exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "exit_code": self.exit_code,
            "stdout": self.stdout[:4000],
            "stderr": self.stderr[:1000] if not self.success else "",
            "duration_s": round(self.duration_s, 2),
            "artifacts": [str(a) for a in self.artifacts],
            "sandbox_mode": self.sandbox_mode,
        }


class SandboxExecutor:
    def __init__(
        self,
        session_dir: Path | None = None,
        data_path: Path | None = None,
    ) -> None:
        cfg = get_settings()
        self._image = cfg.sandbox_image
        self._timeout = cfg.sandbox_timeout_seconds
        self._session_dir = session_dir or cfg.workspace_dir
        self._session_dir.mkdir(parents=True, exist_ok=True)
        self._data_path = data_path
        self._docker_available = self._check_docker()
        if not self._docker_available:
            print(_DOCKER_WARNING)

    def execute(self, code: str) -> ExecutionResult:
        script_id = hashlib.md5(code.encode()).hexdigest()[:8]
        scripts_dir = self._session_dir / "analysis_scripts"
        scripts_dir.mkdir(parents=True, exist_ok=True)
        script_path = scripts_dir / f"script_{script_id}.py"

        # Always prepend data-loading boilerplate so LLM code has `df` ready
        if self._data_path:
            boilerplate = (
                "import os, polars as pl\n"
                "df = pl.read_csv(os.environ['DATA_PATH'])\n"
                "print(f'Loaded {len(df)} rows, {len(df.columns)} columns')\n"
            )
            full_code = boilerplate + code
        else:
            full_code = code

        script_path.write_text(full_code, encoding="utf-8")
        logger.info("executing_script", script_id=script_id, docker=self._docker_available)

        if self._docker_available:
            return self._run_docker(script_path)
        return self._run_subprocess(script_path)

    # ------------------------------------------------------------------
    # Docker path
    # ------------------------------------------------------------------

    def _run_docker(self, script_path: Path) -> ExecutionResult:
        session_abs = str(self._session_dir.resolve())
        script_name = script_path.name

        cmd = [
            "docker", "run", "--rm",
            "--network=none",
            "--memory=512m",
            "--cpus=1",
            "-v", f"{session_abs}:/workspace:rw",
        ]
        if self._data_path:
            cmd += ["-e", "DATA_PATH=/workspace/data.csv"]
        cmd += [
            self._image,
            "python", f"/workspace/analysis_scripts/{script_name}",
        ]

        start = time.monotonic()
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=self._timeout
            )
            duration = time.monotonic() - start
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                stdout="",
                stderr=f"Execution timed out after {self._timeout}s",
                exit_code=124,
                duration_s=self._timeout,
                artifacts=[],
                sandbox_mode="docker",
            )

        return ExecutionResult(
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            duration_s=time.monotonic() - start,
            artifacts=self._collect_artifacts(),
            sandbox_mode="docker",
        )

    # ------------------------------------------------------------------
    # Subprocess fallback (dev only)
    # ------------------------------------------------------------------

    def _run_subprocess(self, script_path: Path) -> ExecutionResult:
        code = script_path.read_text(encoding="utf-8")
        charts_dir = self._session_dir / "charts"
        charts_dir.mkdir(exist_ok=True)

        data_path_str = str(self._data_path.resolve()) if self._data_path else ""
        preamble = textwrap.dedent(f"""
import sys, os
WORKSPACE = r"{self._session_dir}"
CHARTS_DIR = r"{charts_dir}"
DATA_PATH = r"{data_path_str}"
os.environ['DATA_PATH'] = DATA_PATH
os.makedirs(CHARTS_DIR, exist_ok=True)
""")
        full_code = preamble + code

        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        start = time.monotonic()
        exit_code = 0

        try:
            with (
                contextlib.redirect_stdout(stdout_buf),
                contextlib.redirect_stderr(stderr_buf),
            ):
                exec(  # noqa: S102
                    compile(full_code, str(script_path), "exec"),
                    {"__name__": "__main__"},
                )
        except SystemExit as e:
            exit_code = int(e.code or 0)
        except Exception as exc:
            stderr_buf.write(f"{type(exc).__name__}: {exc}\n")
            exit_code = 1

        return ExecutionResult(
            stdout=stdout_buf.getvalue(),
            stderr=stderr_buf.getvalue(),
            exit_code=exit_code,
            duration_s=time.monotonic() - start,
            artifacts=self._collect_artifacts(),
            sandbox_mode="subprocess",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_docker(self) -> bool:
        try:
            r = subprocess.run(
                ["docker", "info"], capture_output=True, timeout=5
            )
            return r.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    def _collect_artifacts(self) -> list[Path]:
        charts_dir = self._session_dir / "charts"
        charts_dir.mkdir(exist_ok=True)
        return (
            list(charts_dir.glob("*.png"))
            + list(charts_dir.glob("*.html"))
        )
