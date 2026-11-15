"""NVCC eligibility, immutable cache consumption, and fallback without CUDA."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).parents[1] / "src/runners/pch" / filename
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load("nvcc_common", "nvcc_common.py")
wrapper = load("kernelbot_nvcc_wrapper", "nvcc.py")
warmer = load("kernelbot_nvcc_warmer", "nvcc_warm.py")
GENERATION = "a" * 64


@pytest.fixture
def cache(tmp_path, monkeypatch):
    for key in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "KERNELBOT_NVCC_PCH_DISABLE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KERNELBOT_NVCC_PCH", "1")
    source = tmp_path / "first.cu"
    source.write_text("#include <torch/types.h>\n#include <cuda.h>\n#include <cuda_runtime.h>\n")
    args = ["-MD", "-MF", "first.d", "-DTORCH_EXTENSION_NAME=first", "-isystem", "/torch",
            "-gencode=arch=compute_75,code=sm_75", "-std=c++20", "-c", str(source), "-o", "first.o"]
    flags = common.compile_flags(args)
    assert flags is not None  # Actual Torch 2.12 prefix, independent of implementation constants.
    fingerprint = tmp_path / "fingerprint"
    fingerprint.write_text(GENERATION)
    monkeypatch.setattr(common, "FINGERPRINT", fingerprint)
    monkeypatch.setenv("KERNELBOT_NVCC_PCH_ROOT", str(tmp_path / "volume"))
    entry = common.cache_root() / GENERATION / common.cache_key(flags, GENERATION)
    (entry / "pch").mkdir(parents=True)
    (entry / "gpumode.cuh").write_text(common.HEADER)
    files = [{"path": f"pch/{name}_ab.pch", "bytes": 4} for name in ("cicc", "cudafe")]
    for item in files:
        (entry / item["path"]).write_bytes(b"stub")
    manifest = common.identity(flags, GENERATION) | {
        "key": entry.name, "profile": "sm75-default", "files": files}
    (entry / "manifest.json").write_text(json.dumps(manifest))
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("KERNELBOT_NVCC_PCH_EVENTS", str(events))
    log = "\n".join(f'using precompiled header file "{entry / item["path"]}"' for item in files)
    calls, responses = [], []

    def run(args, capture=False):
        calls.append((args, capture))
        code, stderr = responses.pop(0) if responses else (0, log)
        return subprocess.CompletedProcess(args, code, "", stderr)

    monkeypatch.setattr(wrapper, "run_compiler", run)
    return SimpleNamespace(**locals())


def test_identity_reuses_modules_but_invalidates_compiler_inputs(cache, monkeypatch):
    c = cache
    other = c.args.copy()
    other[other.index("-DTORCH_EXTENSION_NAME=first")] = "-DTORCH_EXTENSION_NAME=second"
    c.source.write_text(c.source.read_text() + "__global__ void changed_kernel() {}\n")
    assert common.compile_flags(other) == c.flags
    key = common.cache_key(c.flags, GENERATION)
    for flag in ("-O3", "-DUSER=1", "-I/custom", "--use_fast_math", "-std=c++17", "-arch=sm_100"):
        assert common.cache_key(common.compile_flags([*other, flag]), GENERATION) != key
    assert common.cache_key(c.flags, "b" * 64) != key
    monkeypatch.setenv("CPATH", "/different")
    assert common.cache_key(c.flags, GENERATION) != key


@pytest.mark.parametrize("extra", [["@flags.rsp"], ["-include", "custom.h"],
                                    ["--compiler-bindir=/custom"], ["--pch"]])
def test_hidden_options_bypass(cache, extra):
    assert common.compile_flags([*cache.args, *extra]) is None


@pytest.mark.parametrize("prefix", ["#include <cuda_runtime.h>\n", "#define USER_SETTING 1\n"])
def test_header_order_is_preserved(cache, prefix):
    cache.source.write_text(prefix + cache.source.read_text())
    assert common.compile_flags(cache.args) is None


@pytest.mark.parametrize("mode", ["disabled", "missing", "environment"])
def test_passthrough_never_writes_cache(cache, monkeypatch, mode):
    monkeypatch.setenv("KERNELBOT_NVCC_PCH_ROOT", str(cache.tmp_path / "absent"))
    if mode == "disabled":
        monkeypatch.setenv("KERNELBOT_NVCC_PCH_DISABLE", "1")
        monkeypatch.setattr(common, "generation", lambda: pytest.fail("Disabled cache accessed"))
    elif mode == "environment":
        monkeypatch.setenv("NVCC_APPEND_FLAGS", "-include custom.h")
    assert wrapper.main(cache.args) == 0
    assert cache.calls == [(cache.args, False)]
    assert not common.cache_root().exists()


@pytest.mark.parametrize("kind,code,status", [
    ("hit", 0, "hit"), ("partial", 0, "fallback"),
    ("rejected", 1, "fallback"), ("user-error", 2, "error")])
def test_consumption_evidence_and_cache_only_retry(cache, kind, code, status):
    log = {"partial": cache.log.splitlines()[0], "rejected": "invalid precompiled header file"}
    log["user-error"] = cache.log + "\nerror: identifier typo undefined"
    cache.responses.append((code, log.get(kind, cache.log)))
    assert wrapper.main(cache.args) == (0 if kind == "rejected" else code)
    assert cache.calls[0][0][:len(cache.args)] == cache.args
    assert len(cache.calls) == (2 if kind == "rejected" else 1)
    if kind == "rejected":
        assert cache.calls[1] == (cache.args, False)
    event = json.loads(cache.events.read_text())
    assert event["status"] == status
    if status == "hit":
        assert event["frontends"] == ["cicc", "cudafe"]


@pytest.mark.parametrize("change", [
    "truncated", "header", "generation", "escape", "partial", "duplicate", "duplicate-alias"])
def test_bad_ready_metadata_falls_back_before_compiler(cache, change):
    c = cache
    if change == "truncated":
        (c.entry / c.files[0]["path"]).write_bytes(b"bad")
    elif change == "header":
        (c.entry / "gpumode.cuh").write_text("#include <user.h>\n")
    elif change == "generation":
        c.manifest["generation"] = "b" * 64
    elif change == "escape":
        c.files[0]["path"] = "../foreign.pch"
    elif change.startswith("duplicate"):
        path = c.files[0]["path"]
        path = path.replace("pch/", "pch/./") if change.endswith("alias") else path
        c.files.append({"path": path, "bytes": 4})
    else:
        c.files.pop()
    (c.entry / "manifest.json").write_text(json.dumps(c.manifest))
    assert wrapper.main(c.args) == 0 and c.calls == [(c.args, False)]


@pytest.mark.parametrize("count", [2, 3])
def test_validation_does_not_read_gigabyte_payloads(cache, monkeypatch, count):
    if count == 3:
        (cache.entry / "pch/cicc_cd.pch").write_bytes(b"stub")
        cache.files.append({"path": "pch/cicc_cd.pch", "bytes": 4})
        (cache.entry / "manifest.json").write_text(json.dumps(cache.manifest))
    original = Path.open

    def guarded(path, *args, **kwargs):
        assert path.suffix != ".pch"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    assert common.read_manifest(cache.entry, cache.flags, GENERATION)["files"]


@pytest.mark.parametrize("complete", [True, False])
def test_warmer_retains_module_macro_and_publishes_atomically(cache, monkeypatch, complete):
    c = cache
    monkeypatch.setenv("KERNELBOT_NVCC_PCH_ROOT", str(c.tmp_path / "warm"))
    captured = c.flags, common.strip_outputs(c.args, False)
    monkeypatch.setattr(warmer, "capture", lambda *a: captured)

    def compile_header(argv, **kwargs):
        assert "-DTORCH_EXTENSION_NAME=first" in argv
        entry = Path(argv[argv.index("--pch-dir") + 1]).parent
        assert not (entry / "manifest.json").exists()
        for item in c.files[:2 if complete else 1]:
            (entry / item["path"]).write_bytes(b"stub")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(warmer.subprocess, "run", compile_header)
    entry = common.cache_root() / GENERATION / c.entry.name
    if not complete:
        with pytest.raises(ValueError):
            warmer.warm_profile("sm75-default", GENERATION)
        assert not (entry / "manifest.json").exists()
    else:
        result = warmer.warm_profile("sm75-default", GENERATION)
        assert common.read_manifest(entry, c.flags, GENERATION) == result


def test_warmer_accepts_only_curated_profiles(monkeypatch):
    monkeypatch.setattr(warmer, "warm_profile", lambda *a: pytest.fail("Unexpected compilation"))
    with pytest.raises(ValueError, match="fixed profiles"):
        warmer.warm(["-include /user/header.h"])
    assert common.HEADERS == ("torch/types.h", "cuda.h", "cuda_runtime.h")
    for arch, expected in (("75", "7.5"), ("90a", "9.0a"), ("100", "10.0"), ("100a", "10.0a")):
        assert warmer.torch_arch(arch) == expected
