import json
import os
import shlex
import shutil
import subprocess
from typing import Any

from libkernelbot.consts import SubmissionMode
from libkernelbot.utils import KernelBotError, limit_length, setup_logging

logger = setup_logging(__name__)

_TRUE_VALUES = {"1", "true", "yes", "on"}
_DEFAULT_TIMEOUT_SEC = 30
_GUARDED_MODES = frozenset(
    {
        SubmissionMode.BENCHMARK,
        SubmissionMode.PROFILE,
        SubmissionMode.LEADERBOARD,
        SubmissionMode.PRIVATE,
    }
)


class KernelGuardRejected(KernelBotError):
    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result


def _env_enabled(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in _TRUE_VALUES


def should_precheck_submission(mode: SubmissionMode) -> bool:
    return _env_enabled("KERNELGUARD_ENABLED") and mode in _GUARDED_MODES


def _timeout_sec() -> int:
    raw = os.getenv("KERNELGUARD_TIMEOUT_SEC", str(_DEFAULT_TIMEOUT_SEC)).strip()
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("Invalid KERNELGUARD_TIMEOUT_SEC=%r, using %d", raw, _DEFAULT_TIMEOUT_SEC)
        return _DEFAULT_TIMEOUT_SEC


def _profile() -> str | None:
    raw = os.getenv("KERNELGUARD_PROFILE", "").strip()
    return raw or None


def _config_path() -> str | None:
    raw = os.getenv("KERNELGUARD_CONFIG", "").strip()
    return raw or None


def _fail_open_enabled() -> bool:
    return _env_enabled("KERNELGUARD_FAIL_OPEN")


def _default_command() -> list[str]:
    for candidate in ("kernelguard", "kguard"):
        if shutil.which(candidate):
            return [candidate]
    if shutil.which("uvx"):
        return ["uvx", "kernelguard"]
    raise FileNotFoundError("Could not find `kernelguard`, `kguard`, or `uvx` in PATH")


def _command() -> list[str]:
    raw = os.getenv("KERNELGUARD_COMMAND", "").strip()
    if raw:
        return shlex.split(raw)
    return _default_command()


def _analyze_with_cli(code: str) -> dict[str, Any]:
    cmd = [*_command()]
    profile = _profile()
    config_path = _config_path()
    if profile is not None:
        cmd.extend(["--profile", profile])
    if config_path is not None:
        cmd.extend(["--config", config_path])
    cmd.append("--api-mode")

    proc = subprocess.run(
        cmd,
        input=code,
        text=True,
        capture_output=True,
        timeout=_timeout_sec(),
        check=False,
    )
    if proc.returncode != 0:
        stderr = limit_length(proc.stderr.strip(), 300) if proc.stderr else ""
        stdout = limit_length(proc.stdout.strip(), 300) if proc.stdout else ""
        raise RuntimeError(
            "KernelGuard command failed "
            f"(exit={proc.returncode}, stdout={stdout!r}, stderr={stderr!r})"
        )

    lines = [line for line in proc.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("KernelGuard returned no JSON result")

    try:
        result = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"KernelGuard returned invalid JSON: {lines[-1]!r}") from exc

    if not isinstance(result, dict):
        raise RuntimeError("KernelGuard returned a non-object JSON payload")
    return result


def analyze_submission(code: str) -> dict[str, Any]:
    # Always use the single-shot CLI path so KERNELGUARD_TIMEOUT_SEC is enforced.
    return _analyze_with_cli(code)


def enforce_submission_precheck(code: str, file_name: str) -> dict[str, Any] | None:
    if not _env_enabled("KERNELGUARD_ENABLED"):
        return None

    try:
        result = analyze_submission(code)
    except Exception as exc:
        logger.warning("KernelGuard pre-check failed for %s", file_name, exc_info=exc)
        if _fail_open_enabled():
            return None
        raise KernelBotError(
            "KernelGuard pre-check is unavailable right now. Please try again later.",
            code=503,
        ) from exc

    classification = str(result.get("classification", "unknown"))
    if result.get("should_filter"):
        patterns = sorted(
            {
                str(item.get("pattern", "unknown"))
                for item in result.get("matched_patterns", [])
                if isinstance(item, dict)
            }
        )
        reason = str(result.get("filter_reason") or classification)
        details = f"Submission rejected by KernelGuard pre-check ({reason})"
        if patterns:
            details += f". Matched rules: {', '.join(patterns)}"
        raise KernelGuardRejected(details + ".", result=result)

    if classification != "valid":
        logger.info("KernelGuard classified %s as %s", file_name, classification)

    return result
