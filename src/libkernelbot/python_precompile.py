"""Best-effort CPU compilation for ordinary Python submissions."""

import dataclasses
import hashlib
import io
import json
import os
import signal
import subprocess
import tempfile
import time
import zipfile
from contextlib import chdir, contextmanager, suppress
from pathlib import Path

from libkernelbot.consts import Timeout
from libkernelbot.inline_artifacts import prepare_environment
from libkernelbot.run_eval import CPUCompileInfo

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
# Restricted workers cannot upload Modal's large-result blobs. Leave room for metadata.
MAX_TRANSFER_BYTES = 1024 * 1024


def pack_artifacts(root: Path) -> bytes:
    files = [
        path for pattern in ("*/manifest.json", "*/extension.so") for path in root.glob(pattern)
    ]
    if not files:
        return b""
    if sum(path.stat().st_size for path in files) > MAX_ARTIFACT_BYTES:
        raise ValueError("Compiled artifacts exceed the unpacked size limit")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))
    if buffer.tell() > MAX_TRANSFER_BYTES:
        raise ValueError("Compiled artifacts exceed the inline transfer limit")
    return buffer.getvalue()


def write_file(root: Path, name: str, data: bytes):
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Invalid artifact path: {name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def unpack_artifacts(root: Path, encoded: bytes):
    if len(encoded) > MAX_TRANSFER_BYTES:
        raise ValueError("Artifact transfer exceeds the size limit")
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_ARTIFACT_BYTES:
            raise ValueError("Unpacked artifacts exceed the size limit")
        for name in archive.namelist():
            write_file(root, name, archive.read(name))


def should_precompile(config: dict) -> bool:
    return config.get("lang") == "py" and any(
        "load_inline" in source
        for name, source in config.get("sources", {}).items()
        if name.endswith(".py")
    )


def build_identity(config: dict) -> dict:
    return {
        "version": 2,
        "sources": hashlib.sha256(
            json.dumps(config["sources"], sort_keys=True).encode()
        ).hexdigest(),
        "image": Path("/opt/kernelbot-pch/fingerprint").read_text().strip(),
    }


def cuda_arch(sm: str) -> str:
    # task configs use nvcc's SM spelling, e.g. 90a or 100.
    digits = sm.removesuffix("a")
    if not digits.isdigit() or len(digits) < 2:
        raise ValueError(f"Invalid CUDA target: {sm}")
    return f"{int(digits[:-1])}.{sm[len(digits) - 1 :]}"


def _import_submission(work: Path, environment: dict, timeout: float):
    with subprocess.Popen(
        ["python3", "submission.py"],
        cwd=work,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            output, _ = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.communicate()
            raise
    if process.returncode:
        raise RuntimeError(f"CPU import exited {process.returncode}: {output[-1000:]}")


def compile_python(config: dict) -> dict:
    started = time.perf_counter()
    info = CPUCompileInfo(status="skipped")
    packet = {"artifacts": b""}
    try:
        packet["identity"] = build_identity(config)
        with tempfile.TemporaryDirectory(prefix="kernelbot-cpu-") as directory:
            work = Path(directory)
            for name, source in config["sources"].items():
                write_file(work, name, source.encode())
            environment = prepare_environment(work, "capture", cuda_arch(config["arch"]))
            environment["POPCORN_SEED"] = "1"
            _import_submission(work, environment, Timeout.COMPILE)
            packet["artifacts"] = pack_artifacts(work / "artifacts")
            info.artifacts = len(list((work / "artifacts").glob("*/manifest.json")))
            info.status = "compiled" if info.artifacts else "skipped"
    except Exception as exc:
        info.status, info.reason = "fallback", str(exc)[-1000:]
    info.duration = time.perf_counter() - started
    return {**packet, "info": dataclasses.asdict(info)}


@contextmanager
def cpu_artifacts(config: dict):
    """Set up optional reuse; never catch or retry evaluation failures."""
    with tempfile.TemporaryDirectory(prefix="kernelbot-gpu-") as directory:
        work, environment = Path(directory), {}
        info = CPUCompileInfo("fallback")
        try:
            packet = config["cpu_compile"]
            info = CPUCompileInfo(**packet["info"])
            if packet["artifacts"]:
                if packet["identity"] != build_identity(config):
                    raise ValueError("CPU artifact protocol, sources, or runner image changed")
                unpack_artifacts(work / "artifacts", packet["artifacts"])
                environment = prepare_environment(work, "auto", cuda_arch(config["arch"]))
        except Exception as exc:
            info.status, info.reason = "fallback", str(exc)[-1000:]
        previous_environment = dict(os.environ)
        try:
            with chdir(work if environment else Path.cwd()):
                os.environ.update(environment)
                yield info
        finally:
            os.environ.clear()
            os.environ.update(previous_environment)
        events = work / "artifacts" / "events.jsonl"
        if environment:
            for line in events.read_text().splitlines() if events.exists() else []:
                event = json.loads(line)
                info.reused += event["mode"] == "replay"
                info.fallbacks += event["mode"] == "fallback"
                info.reason = event.get("reason", info.reason)
            info.status = "reused" if info.reused else "fallback"
            if info.reused and info.fallbacks:
                info.status = "partial"


def run_with_cpu_artifacts(config: dict, run):
    if config.get("lang") != "py" or not config.get("cpu_compile"):
        return run(config)
    with cpu_artifacts(config) as info:
        result = run(config)
    result.cpu_compile = info
    return result
