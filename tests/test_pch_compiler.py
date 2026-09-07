"""PCH cache isolation and transparent fallback without a local torch install."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "pch_compiler", Path(__file__).parents[1] / "src/runners/pch/compiler.py"
)
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


def command(tmp_path, name="first", extra=()):
    source = tmp_path / f"{name}.cpp"
    source.write_text("#include <torch/extension.h>\nint value() { return 1; }\n")
    return [
        "-MMD",
        "-MF",
        f"{name}.d",
        f"-DTORCH_EXTENSION_NAME={name}",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-isystem",
        "/torch/include",
        "-fPIC",
        "-std=c++17",
        *extra,
        "-c",
        str(source),
        "-o",
        f"{name}.o",
    ]


def test_different_kernels_share_header_key(tmp_path):
    first = compiler.compile_flags(command(tmp_path, "first"))
    second = compiler.compile_flags(command(tmp_path, "second"))
    assert compiler.cache_key(first, "image") == compiler.cache_key(second, "image")


@pytest.mark.parametrize(
    "flags",
    [
        ["-O3"],
        ["-D_GLIBCXX_USE_CXX11_ABI=0"],
        ["-std=c++20"],
        ["-I/other/headers"],
        ["-DOTHER=1"],
        ["-ffast-math"],
    ],
)
def test_incompatible_flags_do_not_reuse(tmp_path, flags):
    normal = compiler.compile_flags(command(tmp_path))
    changed = compiler.compile_flags(command(tmp_path, extra=flags))
    assert compiler.cache_key(normal, "image") != compiler.cache_key(changed, "image")


def test_header_and_environment_invalidation(tmp_path, monkeypatch):
    flags = compiler.compile_flags(command(tmp_path))
    key = compiler.cache_key(flags, "image")
    assert key != compiler.cache_key(flags, "new headers/compiler/torch")
    monkeypatch.setenv("CPLUS_INCLUDE_PATH", "/new/headers")
    assert key != compiler.cache_key(flags, "image")


@pytest.mark.parametrize(
    "source", ["int value() { return 1; }", "#define CUSTOM 1\n#include <torch/extension.h>"]
)
def test_does_not_inject_headers_or_reorder_macros(tmp_path, source):
    args = command(tmp_path)
    Path(args[args.index("-c") + 1]).write_text(source)
    assert compiler.compile_flags(args) is None


@pytest.mark.parametrize(
    "args", [["--version"], ["-v"], ["a.o", "-shared", "-o", "a.so"], ["-o", "a.o", "-c"]]
)
def test_pass_through_non_compilation(args):
    assert compiler.compile_flags(args) is None


def test_miss_is_read_only_and_falls_back(tmp_path, monkeypatch):
    args = command(tmp_path)
    fingerprint = tmp_path / "fingerprint"
    fingerprint.write_text("image")
    cache = tmp_path / "cache"
    monkeypatch.setattr(compiler, "FINGERPRINT", fingerprint)
    monkeypatch.setattr(compiler, "ROOT", cache)
    monkeypatch.delenv("KERNELBOT_PCH_WRITE", raising=False)
    calls = []
    monkeypatch.setattr(compiler, "run_compiler", lambda argv: calls.append(argv) or 0)
    assert compiler.main(args) == 0
    assert calls == [args]
    assert not cache.exists()


def test_hit_injects_pch_without_reusing_kernel_objects(tmp_path, monkeypatch):
    args = command(tmp_path)
    fingerprint = tmp_path / "fingerprint"
    fingerprint.write_text("image")
    monkeypatch.setattr(compiler, "FINGERPRINT", fingerprint)
    monkeypatch.setattr(compiler, "ROOT", tmp_path)
    key = compiler.cache_key(compiler.compile_flags(args), "image")
    entry = tmp_path / key
    entry.mkdir()
    (entry / "torch.h.gch").write_bytes(b"cached pch")
    calls = []
    monkeypatch.setattr(compiler, "run_compiler", lambda argv: calls.append(argv) or 0)
    assert compiler.main(args) == 0
    assert calls == [[*args, "-include", str(entry / "torch.h"), "-Winvalid-pch"]]
