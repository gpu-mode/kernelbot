"""Isolated runtime for problem-owned application validation jobs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

MAX_RESULT_BYTES = 1024 * 1024


def _safe_destination(root: Path, name: str) -> Path:
    destination = (root / name).resolve()
    if destination == root or root not in destination.parents:
        raise ValueError(f"validation source path escapes workspace: {name!r}")
    return destination


def _read_tail(path: Path, limit: int = MAX_RESULT_BYTES) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - limit))
        return handle.read().decode("utf-8", errors="replace")


def _parse_result(stdout_path: Path, returncode: int) -> dict:
    output_lines = [
        line.strip()
        for line in _read_tail(stdout_path).splitlines()
        if line.strip()
    ]
    if not output_lines:
        return {
            "status": "failed",
            "error": f"validation exited {returncode} without a result",
        }
    try:
        result = json.loads(output_lines[-1])
    except json.JSONDecodeError:
        return {
            "status": "failed",
            "error": f"validation exited {returncode} without valid JSON",
        }
    if not isinstance(result, dict):
        return {
            "status": "failed",
            "error": "validation result must be a JSON object",
        }
    if returncode != 0 or result.get("status") == "failed":
        return {
            "status": "failed",
            "error": result.get(
                "failure_reason",
                f"validation exited {returncode}",
            ),
        }
    return {"status": "completed", "result": result}


def run_validation_config(config: dict) -> dict:
    """Execute one validation entrypoint without exposing submission output."""
    sources = config.get("sources")
    main = config.get("main")
    timeout = int(config.get("timeout", 900))
    if not isinstance(sources, dict) or not sources:
        return {"status": "failed", "error": "validation sources are missing"}
    if not isinstance(main, str) or main not in sources:
        return {"status": "failed", "error": "validation main source is missing"}
    if timeout <= 0:
        return {"status": "failed", "error": "validation timeout must be positive"}

    public_config = {
        key: value
        for key, value in config.items()
        if key not in {"sources", "main", "timeout"}
    }
    try:
        with tempfile.TemporaryDirectory(prefix="kernelbot-validation-") as directory:
            root = Path(directory).resolve()
            for name, content in sources.items():
                destination = _safe_destination(root, name)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(content)

            stdout_path = root / "stdout.txt"
            stderr_path = root / "stderr.txt"
            env = os.environ.copy()
            env.update(
                {
                    "KERNELBOT_VALIDATION_CONFIG": json.dumps(public_config),
                    "PYTHONUNBUFFERED": "1",
                    "TORCH_EXTENSIONS_DIR": str(root / "torch-extensions"),
                    "TRITON_CACHE_DIR": str(root / "triton-cache"),
                }
            )
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                try:
                    completed = subprocess.run(
                        [sys.executable, str(_safe_destination(root, main))],
                        cwd=root,
                        env=env,
                        stdout=stdout,
                        stderr=stderr,
                        timeout=timeout,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    return {
                        "status": "failed",
                        "error": f"validation timed out after {timeout} seconds",
                    }

            return _parse_result(stdout_path, completed.returncode)
    except Exception as exc:
        return {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }
