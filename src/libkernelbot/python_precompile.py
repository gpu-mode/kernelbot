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
from contextlib import chdir
from pathlib import Path

from libkernelbot.consts import Timeout
from libkernelbot.inline_artifacts import (
    pack_artifacts,
    prepare_environment,
    read_events,
    unpack_artifacts,
)
from libkernelbot.run_eval import CPUCompileInfo

MAX_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_TRANSFER_BYTES = 1024 * 1024
PROTOCOL_VERSION = 1


def encode_artifacts(artifacts: dict[str, bytes]) -> bytes:
    if not artifacts:
        return b""
    if sum(map(len, artifacts.values())) > MAX_ARTIFACT_BYTES:
        raise ValueError("Compiled artifacts exceed the unpacked size limit")
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, data in artifacts.items():
            archive.writestr(name, data)
    encoded = buffer.getvalue()
    # Restricted Modal workers cannot create large-result blobs. Stay below the
    # SDK's 2 MiB inline result limit, including metadata and serialization.
    if len(encoded) > MAX_TRANSFER_BYTES:
        raise ValueError("Compiled artifacts exceed the inline transfer limit")
    return encoded


def decode_artifacts(encoded: bytes) -> dict[str, bytes]:
    if len(encoded) > MAX_TRANSFER_BYTES:
        raise ValueError("Artifact transfer exceeds the size limit")
    with zipfile.ZipFile(io.BytesIO(encoded)) as archive:
        if sum(item.file_size for item in archive.infolist()) > MAX_ARTIFACT_BYTES:
            raise ValueError("Unpacked artifacts exceed the size limit")
        return {name: archive.read(name) for name in archive.namelist()}


def should_precompile(config: dict) -> bool:
    return config.get("lang") == "py" and any(
        "load_inline" in source
        for name, source in config.get("sources", {}).items()
        if name.endswith(".py")
    )


def source_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(config["sources"], sort_keys=True).encode()).hexdigest()


def image_fingerprint() -> str:
    return Path("/opt/kernelbot-pch/fingerprint").read_text().strip()


def cuda_arch(sm: str) -> str:
    # task configs use nvcc's SM spelling, e.g. 90a or 100.
    suffix = "a" if sm.endswith("a") else ""
    digits = sm.removesuffix("a")
    if not digits.isdigit() or len(digits) < 2:
        raise ValueError(f"Invalid CUDA target: {sm}")
    return f"{int(digits[:-1])}.{digits[-1]}{suffix}"


def _import_submission(work: Path, environment: dict, timeout: float):
    with subprocess.Popen(
        ["python3", "submission.py"],
        cwd=work,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    ) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
    if process.returncode:
        raise RuntimeError(f"CPU import exited {process.returncode}: {(stderr or stdout)[-1000:]}")


def compile_python(config: dict) -> dict:
    started = time.perf_counter()
    info = CPUCompileInfo(status="skipped")
    packet = {"version": PROTOCOL_VERSION, "artifacts": b"", "info": {}}
    try:
        if not should_precompile(config):
            return packet
        import torch

        if torch.cuda.is_available() or torch.cuda.device_count():
            raise RuntimeError("CPU compilation worker has a visible GPU")
        packet.update(source_hash=source_hash(config), image_fingerprint=image_fingerprint())
        with tempfile.TemporaryDirectory(prefix="kernelbot-cpu-") as directory:
            work = Path(directory)
            for name, text in config["sources"].items():
                path = work / name
                if not path.resolve().is_relative_to(work):
                    raise ValueError(f"Invalid source path: {name}")
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            environment = prepare_environment(work, "capture", cuda_arch(config["arch"]))
            environment["POPCORN_SEED"] = "1"
            _import_submission(work, environment, Timeout.COMPILE)
            artifacts = pack_artifacts(work / "artifacts")
            packet["artifacts"] = encode_artifacts(artifacts)
            info.artifacts = sum(name.endswith("/manifest.json") for name in artifacts)
            info.status = "compiled" if info.artifacts else "skipped"
            if not info.artifacts:
                info.reason = "No import-time load_inline library"
    except Exception as exc:
        info.status, info.reason = "fallback", str(exc)[-1000:]
        packet["artifacts"] = b""
    finally:
        info.duration = time.perf_counter() - started
        packet["info"] = dataclasses.asdict(info)
    return packet


def run_with_cpu_artifacts(config: dict, run):
    packet = config.get("cpu_compile")
    if config.get("lang") != "py" or not packet:
        return run(config)
    try:
        info = CPUCompileInfo(**packet["info"])
    except (KeyError, TypeError):
        result = run(config)
        result.cpu_compile = CPUCompileInfo("fallback", reason="Invalid CPU build metadata")
        return result
    if not packet.get("artifacts"):
        result = run(config)
        result.cpu_compile = info
        return result

    with tempfile.TemporaryDirectory(prefix="kernelbot-gpu-") as directory:
        work = Path(directory)
        try:
            if packet["version"] != PROTOCOL_VERSION:
                raise ValueError("CPU artifact protocol changed")
            if packet["source_hash"] != source_hash(config):
                raise ValueError("Submission sources changed after CPU compilation")
            if packet["image_fingerprint"] != image_fingerprint():
                raise ValueError("CPU and GPU runner images differ")
            unpack_artifacts(work / "artifacts", decode_artifacts(packet["artifacts"]))
            environment = prepare_environment(work, "auto", cuda_arch(config["arch"]))
        except Exception as exc:
            info.status, info.reason = "fallback", str(exc)[-1000:]
            result = run(config)
            result.cpu_compile = info
            return result
        previous_environment = dict(os.environ)
        try:
            with chdir(work):
                os.environ.update(environment)
                result = run(config)
        finally:
            os.environ.clear()
            os.environ.update(previous_environment)
        events = read_events(work / "artifacts")
        info.reused = sum(event["mode"] == "replay" for event in events)
        info.fallbacks = sum(event["mode"] == "fallback" for event in events)
        info.status = "reused" if info.reused else "fallback"
        if info.reused and info.fallbacks:
            info.status = "partial"
        reasons = [event["reason"] for event in events if event["mode"] == "fallback"]
        info.reason = reasons[-1] if reasons else ""
        result.cpu_compile = info
        return result
