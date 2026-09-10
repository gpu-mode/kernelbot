import asyncio
import json
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from libkernelbot.consts import get_gpu_by_name
from libkernelbot.inline_artifacts import _check_dependencies, _dependencies
from libkernelbot.launchers.modal import ModalLauncher
from libkernelbot.python_precompile import (
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


@pytest.mark.parametrize("changed", ["sources", "image", "version"])
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
