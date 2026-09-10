"""CPU-only adapter tests; Modal integration tests exercise real library loading."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from libkernelbot.inline_artifacts import install


@pytest.fixture
def extension(tmp_path, monkeypatch):
    binary = tmp_path / "compiled.so"
    binary.write_bytes(b"fake compiled extension")
    builds = Mock(return_value=SimpleNamespace(__name__="example", __file__=str(binary)))

    def original(name, cpp_sources, cuda_sources=None, extra_cuda_cflags=None, extra_ldflags=None):
        return builds(name, cpp_sources, cuda_sources, extra_cuda_cflags, extra_ldflags)

    extension = SimpleNamespace(load_inline=original, builds=builds)
    torch = SimpleNamespace(
        __version__="test",
        version=SimpleNamespace(cuda="13.3"),
        _C=SimpleNamespace(_GLIBCXX_USE_CXX11_ABI=True),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch.utils", SimpleNamespace(cpp_extension=extension))
    monkeypatch.setenv("KERNELBOT_INLINE_WORKDIR", str(tmp_path))
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "7.5")
    monkeypatch.setenv("KERNELBOT_INLINE_MODE", "capture")
    with patch("libkernelbot.inline_artifacts._dependencies", return_value={}):
        install()
        extension.load_inline("example", "source", cuda_sources="cuda")
    extension.load_inline = original
    monkeypatch.setenv("KERNELBOT_INLINE_MODE", "auto")
    install()
    return extension


def test_loads_saved_module_once_without_compiling(extension, tmp_path):
    module, spec = SimpleNamespace(), Mock()
    with (
        patch("importlib.util.spec_from_file_location", return_value=spec) as make_spec,
        patch("importlib.util.module_from_spec", return_value=module),
    ):
        for _ in range(2):
            assert extension.load_inline("example", "source", cuda_sources="cuda") is module
    assert make_spec.call_args.args[0] == "example"
    assert make_spec.call_args.args[1].read_bytes() == (tmp_path / "compiled.so").read_bytes()
    spec.loader.exec_module.assert_called_once_with(module)
    assert extension.builds.call_count == 1


@pytest.mark.parametrize("change", ["source", "flags", "arch", "manifest", "binary", "linker"])
def test_incompatible_artifacts_fall_back_before_loading(extension, tmp_path, monkeypatch, change):
    arguments = {"name": "example", "cpp_sources": "source", "cuda_sources": "cuda"}
    manifest_path = next((tmp_path / "artifacts").glob("*/manifest.json"))
    if change == "source":
        arguments["cpp_sources"] = "changed"
    elif change == "flags":
        arguments["extra_cuda_cflags"] = ["-O3"]
    elif change == "arch":
        monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "9.0a")
    elif change == "manifest":
        manifest = json.loads(manifest_path.read_text())
        manifest["key"] = "different"
        manifest_path.write_text(json.dumps(manifest))
    elif change == "binary":
        manifest_path.with_name("extension.so").write_bytes(b"corrupted")
    else:
        arguments["extra_ldflags"] = ["-lcustom"]
    with patch("importlib.util.spec_from_file_location") as load:
        extension.load_inline(**arguments)
        load.assert_not_called()
    assert extension.builds.call_count == 2
    event = json.loads((tmp_path / "artifacts/events.jsonl").read_text())
    assert event["mode"] == "fallback" and event["reason"]
