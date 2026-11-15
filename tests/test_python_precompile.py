import asyncio
import hashlib
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libkernelbot.consts import get_gpu_by_name
from libkernelbot.inline_artifacts import (
    _check_dependencies,
    _dependencies,
    _dependency_digest,
    nvcc_fingerprint,
)
from libkernelbot.launchers.modal import ModalLauncher
from libkernelbot.python_precompile import (
    build_identity,
    compile_python,
    cuda_arch,
    pack_artifacts,
    run_with_cpu_artifacts,
    should_precompile,
    unpack_artifacts,
    write_file,
)
from libkernelbot.run_eval import FullResult, SystemInfo


@pytest.fixture
def config():
    return {
        "lang": "py",
        "arch": "75",
        "main": "eval.py",
        "mode": "test",
        "sources": {
            "submission.py": "from torch.utils.cpp_extension import load_inline",
            "eval.py": "",
        },
    }


@pytest.fixture
def result():
    return FullResult(success=True, error="", system=SystemInfo())


@pytest.mark.parametrize("sm,expected", [("75", "7.5"), ("90a", "9.0a"), ("100", "10.0")])
def test_target_architecture(sm, expected):
    assert cuda_arch(sm) == expected


def test_only_python_load_inline_submissions_are_candidates(config):
    assert should_precompile(config)
    assert not should_precompile({**config, "lang": "cu"})
    assert not should_precompile({**config, "sources": {"submission.py": "import triton"}})


@pytest.mark.asyncio
async def test_launcher_compiles_before_gpu_and_leaves_input_config_unchanged(config, result):
    cpu, gpu = MagicMock(), MagicMock()
    packet = {"artifacts": {"module": b"binary"}, "info": {"status": "compiled"}}
    cpu.remote.aio = AsyncMock(return_value=packet)
    gpu.remote.aio = AsyncMock(return_value=result)
    order = []

    def lookup(app, name):
        assert app == "debug-runner"
        order.append(name)
        if name == "compile_python_submission":
            return cpu
        cpu.remote.aio.assert_awaited_once_with(config=config)
        return gpu

    with patch("modal.Function.from_name", side_effect=lookup):
        actual = await ModalLauncher([], app_name="debug-runner").run_submission(
            config,
            get_gpu_by_name("T4"),
            AsyncMock(),
        )
    assert actual is result
    assert order == ["compile_python_submission", "run_pytorch_script_t4"]
    assert "cpu_compile" not in config
    gpu.remote.aio.assert_awaited_once_with(config={**config, "cpu_compile": packet})


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError("old deployment"), TimeoutError("CPU timed out")])
async def test_cpu_service_failure_still_runs_gpu(config, result, error):
    cpu, gpu = MagicMock(), MagicMock()
    cpu.remote.aio = AsyncMock(side_effect=error)
    gpu.remote.aio = AsyncMock(return_value=result)
    with patch("modal.Function.from_name", side_effect=[cpu, gpu]):
        actual = await ModalLauncher([]).run_submission(config, get_gpu_by_name("T4"), AsyncMock())
    assert actual is result
    assert gpu.remote.aio.call_args.kwargs["config"]["cpu_compile"]["info"]["status"] == "fallback"


@pytest.mark.asyncio
async def test_cancellation_does_not_launch_gpu(config):
    cpu = MagicMock()
    cpu.remote.aio = AsyncMock(side_effect=asyncio.CancelledError())
    with patch("modal.Function.from_name", return_value=cpu) as lookup:
        with pytest.raises(asyncio.CancelledError):
            await ModalLauncher([]).run_submission(config, get_gpu_by_name("T4"), AsyncMock())
    assert lookup.call_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("disabled", [True, False])
async def test_ordinary_python_or_operator_disable_bypasses_cpu(
    config, result, monkeypatch, disabled
):
    if disabled:
        monkeypatch.setenv("KERNELBOT_CPU_COMPILE", "0")
    else:
        config["sources"]["submission.py"] = "import torch"
    gpu = MagicMock()
    gpu.remote.aio = AsyncMock(return_value=result)
    with patch("modal.Function.from_name", return_value=gpu) as lookup:
        await ModalLauncher([]).run_submission(config, get_gpu_by_name("T4"), AsyncMock())
    lookup.assert_called_once_with("discord-bot-runner", "run_pytorch_script_t4")
    gpu.remote.aio.assert_awaited_once_with(config=config)


@pytest.mark.parametrize(
    "error", [RuntimeError("CUDA unavailable"), subprocess.TimeoutExpired("python", 1)]
)
def test_cpu_import_failures_return_fallback(config, monkeypatch, error):
    with (
        patch("libkernelbot.python_precompile.build_identity", return_value={"image": "image"}),
        patch("libkernelbot.python_precompile._import_submission", side_effect=error),
    ):
        packet = compile_python(config)
    assert packet["info"]["status"] == "fallback"
    assert packet["artifacts"] == b""


@pytest.mark.parametrize("changed", ["sources", "image", "version", "nvcc_toolchain"])
def test_gpu_rejects_stale_build_without_failing_submission(config, result, changed):
    packet = {
        "identity": {"image": "image"},
        "artifacts": {"unused.so": b"binary"},
        "info": {"status": "compiled"},
    }
    packet["identity"][changed] = "different"
    run = MagicMock(return_value=result)
    with patch("libkernelbot.python_precompile.build_identity", return_value={"image": "image"}):
        actual = run_with_cpu_artifacts({**config, "cpu_compile": packet}, run)
    assert actual is result and actual.cpu_compile.status == "fallback"
    run.assert_called_once()


@pytest.mark.parametrize("success", [True, False])
def test_gpu_context_restores_environment_and_records_reuse(config, result, success, tmp_path):
    write_file(tmp_path, "key/extension.so", b"binary")
    result.success = success
    config["cpu_compile"] = {
        "identity": {"image": "image"},
        "artifacts": pack_artifacts(tmp_path),
        "info": {"status": "compiled"},
    }
    previous_cwd, previous_environment = Path.cwd(), dict(os.environ)

    def run(_):
        assert os.environ["KERNELBOT_INLINE_MODE"] == "auto"
        root = Path(os.environ["KERNELBOT_INLINE_WORKDIR"]) / "artifacts"
        (root / "events.jsonl").write_text(json.dumps({"mode": "replay"}) + "\n")
        return result

    with patch("libkernelbot.python_precompile.build_identity", return_value={"image": "image"}):
        actual = run_with_cpu_artifacts(config, run)
    assert actual.cpu_compile.status == "reused" and actual.cpu_compile.reused == 1
    assert actual.success == success
    assert Path.cwd() == previous_cwd and dict(os.environ) == previous_environment


def test_compressed_transport_roundtrip(tmp_path):
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    artifacts = {"key/extension.so": b"binary" * 1000, "key/manifest.json": b"{}"}
    for name, data in {**artifacts, "build/main.o": b"exclude"}.items():
        write_file(cpu, name, data)
    unpack_artifacts(gpu, pack_artifacts(cpu))
    assert {
        str(p.relative_to(gpu)): p.read_bytes() for p in gpu.rglob("*") if p.is_file()
    } == artifacts


def test_transport_size_limit_uses_fallback_instead_of_large_modal_blobs(tmp_path, monkeypatch):
    monkeypatch.setattr("libkernelbot.python_precompile.MAX_TRANSFER_BYTES", 1)
    write_file(tmp_path, "key/extension.so", b"binary")
    with pytest.raises(ValueError, match="inline transfer limit"):
        pack_artifacts(tmp_path)


def test_transfer_rejects_path_escape(tmp_path):
    import io
    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("../escape.so", b"binary")
    with pytest.raises(ValueError, match="Invalid artifact path"):
        unpack_artifacts(tmp_path, buffer.getvalue())


def test_header_dependencies_relocate_and_detect_changes(tmp_path, monkeypatch):
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    build = cpu / "build"
    build.mkdir(parents=True)
    gpu.mkdir()
    (cpu / "kernel.h").write_text("#define VALUE 1")
    (gpu / "kernel.h").write_text("#define VALUE 1")
    monkeypatch.setenv("KERNELBOT_INLINE_WORKDIR", str(cpu))
    output = f"main.o: #deps 2\n    {build / 'main.cpp'}\n    {cpu / 'kernel.h'}\n"
    with patch(
        "libkernelbot.inline_artifacts.subprocess.run", return_value=SimpleNamespace(stdout=output)
    ):
        dependencies = _dependencies(build, cpu)
    assert len(dependencies) == 1
    monkeypatch.setenv("KERNELBOT_INLINE_WORKDIR", str(gpu))
    _check_dependencies(dependencies, gpu)
    (gpu / "kernel.h").write_text("#define VALUE 2")
    with pytest.raises(RuntimeError, match="Build dependency changed"):
        _check_dependencies(dependencies, gpu)


def test_missing_dependency_information_cannot_be_reused(tmp_path, monkeypatch):
    monkeypatch.setenv("KERNELBOT_INLINE_WORKDIR", str(tmp_path))
    with pytest.raises(RuntimeError, match="no compiler dependency"):
        _check_dependencies(None, tmp_path)


def test_cpu_packet_identity_tracks_experimental_compiler(config, tmp_path, monkeypatch):
    fingerprint = tmp_path / "nvcc-fingerprint"
    monkeypatch.setattr("libkernelbot.inline_artifacts.NVCC_FINGERPRINT", fingerprint)
    with patch("libkernelbot.python_precompile.Path.read_text", return_value="host headers"):
        original = build_identity(config)
        assert original["nvcc_toolchain"] is None
        fingerprint.write_bytes(b"first compiler and wrapper digest\n")
        first = build_identity(config)
        fingerprint.write_bytes(b"second compiler and wrapper digest\n")
        second = build_identity(config)
    assert first["image"] == second["image"] == original["image"]
    assert first["sources"] == second["sources"] == original["sources"]
    assert len({first["nvcc_toolchain"], second["nvcc_toolchain"], None}) == 3


def test_empty_nvcc_fingerprint_cannot_authorize_reuse(tmp_path, monkeypatch):
    fingerprint = tmp_path / "nvcc-fingerprint"
    fingerprint.write_bytes(b"\n")
    monkeypatch.setattr("libkernelbot.inline_artifacts.NVCC_FINGERPRINT", fingerprint)
    with pytest.raises(RuntimeError, match="fingerprint is empty"):
        nvcc_fingerprint()


@pytest.fixture
def trusted_pch(tmp_path, monkeypatch):
    generation, key = "a" * 64, "b" * 64
    root = tmp_path / "trusted-cache"
    entry = root / generation / key
    (entry / "pch").mkdir(parents=True)
    header = entry / "gpumode.cuh"
    header.write_text("#include <ATen/ATen.h>\n#pragma hdrstop\n")
    files = []
    for frontend in ("cicc", "cudafe"):
        name = f"pch/{frontend}_123456.pch"
        (entry / name).write_bytes(b"compiler cache contents")
        files.append({"path": name, "bytes": (entry / name).stat().st_size})
    manifest = {
        "version": 1,
        "generation": generation,
        "key": key,
        "profile": "sm75-default",
        "flags": ["-std=c++20"],
        "environment": {},
        "header_sha256": hashlib.sha256(header.read_bytes()).hexdigest(),
        "files": files,
    }
    (entry / "manifest.json").write_text(json.dumps(manifest))
    fingerprint = tmp_path / "nvcc-fingerprint"
    fingerprint.write_text(generation + "\n")
    monkeypatch.setattr("libkernelbot.inline_artifacts.NVCC_FINGERPRINT", fingerprint)
    monkeypatch.setattr("libkernelbot.inline_artifacts.NVCC_PCH_ROOT", root)
    return entry, manifest, fingerprint


def test_trusted_pch_dependencies_do_not_read_cache_bodies(trusted_pch, tmp_path, monkeypatch):
    entry, _, _ = trusted_pch
    original_open = Path.open

    def small_files_only(path, *args, **kwargs):
        assert path.suffix != ".pch", "Artifact replay must not read multi-GB PCH bodies"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", small_files_only)
    cpu, gpu = tmp_path / "cpu", tmp_path / "gpu"
    build = cpu / "build"
    build.mkdir(parents=True)
    gpu.mkdir()
    for work in (cpu, gpu):
        (work / "user.h").write_text("#define VALUE 1\n")
    pch = entry / "pch/cicc_123456.pch"
    output = f"cuda.o: #deps 2\n    {pch}\n    {cpu / 'user.h'}\n"
    with patch(
        "libkernelbot.inline_artifacts.subprocess.run", return_value=SimpleNamespace(stdout=output)
    ):
        dependencies = _dependencies(build, cpu)
    assert dependencies[str(pch)].startswith("nvcc-pch-v1:")
    assert len(dependencies) == 2
    _check_dependencies(dependencies, gpu)
    (gpu / "user.h").write_text("#define VALUE 2\n")
    with pytest.raises(RuntimeError, match="Build dependency changed"):
        _check_dependencies(dependencies, gpu)


@pytest.mark.parametrize("change", ["manifest", "header", "generation", "size", "missing"])
def test_trusted_pch_dependency_changes_reject_replay(trusted_pch, tmp_path, change):
    entry, manifest, fingerprint = trusted_pch
    path = entry / "pch/cicc_123456.pch"
    dependencies = {str(path): _dependency_digest(path)}
    if change == "manifest":
        manifest["flags"] = ["-std=c++20", "-O3"]
        (entry / "manifest.json").write_text(json.dumps(manifest))
    elif change == "header":
        (entry / "gpumode.cuh").write_text("#include <different.h>\n")
    elif change == "generation":
        fingerprint.write_text("c" * 64 + "\n")
    elif change == "size":
        path.write_bytes(b"truncated")
    else:
        (entry / "manifest.json").unlink()
    with pytest.raises((RuntimeError, FileNotFoundError)):
        _check_dependencies(dependencies, tmp_path)


def test_untrusted_pch_is_not_skipped_or_read(tmp_path, monkeypatch):
    path = tmp_path / "user.pch"
    path.write_bytes(b"not trusted")
    monkeypatch.setenv("KERNELBOT_NVCC_PCH_ROOT", str(tmp_path))
    with patch.object(Path, "open", side_effect=AssertionError("Must not read PCH contents")):
        with pytest.raises(RuntimeError, match="outside the trusted"):
            _dependency_digest(path)


def test_pch_file_must_be_named_by_the_complete_ready_manifest(trusted_pch):
    entry, _, _ = trusted_pch
    path = entry / "pch/unknown.pch"
    path.write_bytes(b"unlisted cache")
    with pytest.raises(RuntimeError, match="not covered"):
        _dependency_digest(path)
