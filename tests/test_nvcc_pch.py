"""NVCC cache eligibility, immutable identity, and failure recovery without CUDA."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SOURCE = Path(__file__).parents[1] / "src/runners/pch"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load("nvcc_common", SOURCE / "nvcc_common.py")
wrapper = load("kernelbot_nvcc_wrapper", SOURCE / "nvcc.py")
warmer = load("kernelbot_nvcc_warmer", SOURCE / "nvcc_warm.py")
GENERATION = "a" * 64


@pytest.fixture(autouse=True)
def clean_flags(monkeypatch):
    for key in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS", "KERNELBOT_NVCC_PCH_DISABLE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("KERNELBOT_NVCC_PCH", "1")


def command(tmp_path, name="first", extra=()):
    source = tmp_path / f"{name}.cu"
    prefix = "".join(f"#include <{h}>\n" for h in common.HEADERS)
    source.write_text(prefix + '#include "user.h"\n__global__ void kernel() {}\n')
    return [
        "-MD",
        "-MF",
        f"{name}.d",
        f"-DTORCH_EXTENSION_NAME={name}",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-isystem",
        "/torch/include",
        "-gencode=arch=compute_75,code=sm_75",
        "--compiler-options",
        "-fPIC",
        "-std=c++20",
        *extra,
        "-c",
        str(source),
        "-o",
        f"{name}.o",
    ]


def ready(tmp_path, monkeypatch, args):
    root = tmp_path / "volume"
    monkeypatch.setenv("KERNELBOT_NVCC_PCH_ROOT", str(root))
    fingerprint = tmp_path / "fingerprint"
    fingerprint.write_text(GENERATION)
    monkeypatch.setattr(common, "FINGERPRINT", fingerprint)
    flags = common.compile_flags(args)
    key = common.cache_key(flags, GENERATION)
    entry = root / GENERATION / key
    (entry / "pch").mkdir(parents=True)
    (entry / "gpumode.cuh").write_text(common.HEADER)
    files = []
    for name in ("cicc_ab_12.pch", "cudafe_ab_12.pch"):
        path = entry / "pch" / name
        path.write_bytes(b"PCH stub")
        files.append({"path": "pch/" + name, "bytes": path.stat().st_size})
    manifest = common.identity(flags, GENERATION) | {
        "key": key,
        "profile": "sm75-default",
        "files": files,
    }
    (entry / "manifest.json").write_text(json.dumps(manifest))
    return entry, manifest


def fake_compiler(monkeypatch, responses):
    calls = []

    def run(args, capture=False):
        calls.append((args, capture))
        code, stderr = responses[len(calls) - 1]
        return subprocess.CompletedProcess(args, code, "", stderr)

    monkeypatch.setattr(wrapper, "run_compiler", run)
    return calls


def hit_log(entry):
    return "\n".join(
        f'using precompiled header file "{path}"' for path in sorted((entry / "pch").glob("*.pch"))
    )


def test_different_modules_and_user_sources_share_only_header_profile(tmp_path):
    first = command(tmp_path)
    second = command(tmp_path, "second")
    source = Path(second[second.index("-c") + 1])
    source.write_text(source.read_text().replace("kernel() {}", "kernel() { int changed = 2; }"))
    one, two = common.compile_flags(first), common.compile_flags(second)
    assert common.cache_key(one, GENERATION) == common.cache_key(two, GENERATION)
    assert "-DTORCH_EXTENSION_NAME=first" in first


def test_pytorch_212_cuda_prefix_is_eligible_without_extra_headers(tmp_path):
    args = command(tmp_path)
    Path(args[args.index("-c") + 1]).write_text(
        "#include <torch/types.h>\n"
        "#include <cuda.h>\n"
        "#include <cuda_runtime.h>\n"
        "__global__ void kernel() {}\n"
    )
    assert common.compile_flags(args) is not None
    assert common.HEADERS == ("torch/types.h", "cuda.h", "cuda_runtime.h")


@pytest.mark.parametrize(
    "extra",
    [
        ["-O3"],
        ["-DUSER_MACRO=1"],
        ["-I/custom/headers"],
        ["--use_fast_math"],
        ["-std=c++17"],
        ["-gencode=arch=compute_100,code=sm_100"],
    ],
)
def test_flags_must_match_exactly(tmp_path, extra):
    baseline = common.compile_flags(command(tmp_path))
    changed = common.compile_flags(command(tmp_path, "changed", extra))
    assert common.cache_key(baseline, GENERATION) != common.cache_key(changed, GENERATION)


def test_toolchain_and_environment_invalidate_profile(tmp_path, monkeypatch):
    flags = common.compile_flags(command(tmp_path))
    key = common.cache_key(flags, GENERATION)
    assert key != common.cache_key(flags, "b" * 64)
    monkeypatch.setenv("CPATH", "/different/headers")
    assert key != common.cache_key(flags, GENERATION)


@pytest.mark.parametrize(
    "prefix",
    [
        "#include <cuda_runtime.h>\n",
        "#define USER_SETTING 1\n",
        "#include <ATen/ATen.h> #include <torch/types.h>\n",
    ],
)
def test_no_implicit_headers_and_macro_reordering_bypass(tmp_path, prefix):
    args = command(tmp_path)
    Path(args[args.index("-c") + 1]).write_text(prefix + "__global__ void kernel() {}\n")
    assert common.compile_flags(args) is None


@pytest.mark.parametrize(
    "extra", [["@flags.rsp"], ["-include", "custom.h"], ["--compiler-bindir=/custom"], ["--pch"]]
)
def test_custom_compiler_and_hidden_flags_bypass(tmp_path, extra):
    assert common.compile_flags(command(tmp_path, extra=extra)) is None


def test_environment_injected_options_bypass(tmp_path, monkeypatch):
    monkeypatch.setenv("NVCC_APPEND_FLAGS", "-include /user/header.h")
    assert common.compile_flags(command(tmp_path)) is None


def test_unknown_profile_never_creates_volume_files(tmp_path, monkeypatch):
    args = command(tmp_path)
    monkeypatch.setenv("KERNELBOT_NVCC_PCH_ROOT", str(tmp_path / "absent"))
    monkeypatch.setattr(common, "generation", lambda: GENERATION)
    calls = fake_compiler(monkeypatch, [(0, "")])
    assert wrapper.main(args) == 0
    assert calls == [(args, False)]
    assert not (tmp_path / "absent").exists()


def test_hit_keeps_original_flags_and_reports_both_frontends(tmp_path, monkeypatch):
    args = command(tmp_path)
    entry, _ = ready(tmp_path, monkeypatch, args)
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("KERNELBOT_NVCC_PCH_EVENTS", str(events))
    calls = fake_compiler(monkeypatch, [(0, hit_log(entry))])
    assert wrapper.main(args) == 0
    assert calls[0][0][: len(args)] == args
    assert calls[0][0][-1] == "--pch-verbose"
    record = json.loads(events.read_text())
    assert record["status"] == "hit" and record["frontends"] == ["cicc", "cudafe"]


def test_rejected_pch_retries_once_without_injected_header(tmp_path, monkeypatch):
    args = command(tmp_path)
    ready(tmp_path, monkeypatch, args)
    calls = fake_compiler(monkeypatch, [(1, "invalid precompiled header file"), (0, "")])
    assert wrapper.main(args) == 0
    assert len(calls) == 2 and calls[1] == (args, False)


def test_user_compile_error_is_not_retried(tmp_path, monkeypatch):
    args = command(tmp_path)
    entry, _ = ready(tmp_path, monkeypatch, args)
    calls = fake_compiler(
        monkeypatch, [(2, hit_log(entry) + '\nerror: identifier "typo" undefined')]
    )
    assert wrapper.main(args) == 2
    assert len(calls) == 1


def test_partial_pch_consumption_is_not_claimed_as_hit(tmp_path, monkeypatch):
    args = command(tmp_path)
    entry, _ = ready(tmp_path, monkeypatch, args)
    fake_compiler(monkeypatch, [(0, hit_log(entry).splitlines()[0])])
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("KERNELBOT_NVCC_PCH_EVENTS", str(events))
    assert wrapper.main(args) == 0
    assert json.loads(events.read_text())["status"] == "fallback"


@pytest.mark.parametrize("change", ["truncated", "header", "generation", "escape", "partial"])
def test_invalid_ready_metadata_falls_back_before_compiler(tmp_path, monkeypatch, change):
    args = command(tmp_path)
    entry, manifest = ready(tmp_path, monkeypatch, args)
    if change == "truncated":
        (entry / manifest["files"][0]["path"]).write_bytes(b"broken")
    elif change == "header":
        (entry / "gpumode.cuh").write_text("#include <user.h>\n")
    elif change == "generation":
        manifest["generation"] = "b" * 64
    elif change == "escape":
        manifest["files"][0]["path"] = "../foreign.pch"
    else:
        manifest["files"].pop()
    (entry / "manifest.json").write_text(json.dumps(manifest))
    calls = fake_compiler(monkeypatch, [(0, "")])
    assert wrapper.main(args) == 0
    assert calls == [(args, False)]


def test_kill_switch_does_not_access_cache(tmp_path, monkeypatch):
    args = command(tmp_path)
    monkeypatch.setenv("KERNELBOT_NVCC_PCH_DISABLE", "1")
    monkeypatch.setattr(common, "generation", lambda: pytest.fail("Cache accessed while disabled"))
    calls = fake_compiler(monkeypatch, [(0, "")])
    assert wrapper.main(args) == 0
    assert calls == [(args, False)]


def test_warmer_rejects_untrusted_selectors_before_any_compile(monkeypatch):
    monkeypatch.setattr(warmer, "warm_profile", lambda *args: pytest.fail("Unexpected compilation"))
    with pytest.raises(ValueError, match="fixed profiles"):
        warmer.warm(["-include /user/header.h"])


def test_profile_catalog_has_bounded_architecture_and_flag_sets():
    assert len(common.PROFILES) == len(common.DEFAULT_PROFILES) == 18
    assert set(common.DEFAULT_PROFILES) == set(common.PROFILES)
    assert common.ARCHES == ("75", "80", "89", "90a", "100")
    assert list(common.FLAG_PROFILES.values()) == [[], ["-O3"], ["-O3", "--use_fast_math"]]
    assert common.PROFILES["sm90a-default"] == ("90a", [])
    assert common.PROFILES["sm100-O3-fast-math"] == ("100", ["-O3", "--use_fast_math"])
    assert "cutlass" not in common.HEADER.lower()


def test_observed_explicit_architecture_profiles_preserve_exact_ordered_flags():
    assert common.EXPLICIT_PROFILES == {
        "sm100a-O3-fast-math-arch": ("100a", ["-O3", "--use_fast_math", "-arch=sm_100a"]),
        "sm100-O3-arch-fast-math": ("100", ["-O3", "-arch=sm_100", "--use_fast_math"]),
        "sm80-O3-cxx17-arch-fast-math": (
            "80",
            ["-O3", "-std=c++17", "-arch=sm_80", "--use_fast_math"],
        ),
    }


@pytest.mark.parametrize(
    ("arch", "expected"), [("75", "7.5"), ("90a", "9.0a"), ("100", "10.0"), ("100a", "10.0a")]
)
def test_capture_uses_curated_architecture_including_blackwell_suffix(arch, expected):
    assert warmer.torch_arch(arch) == expected


def test_ready_validation_never_reads_pch_payloads(tmp_path, monkeypatch):
    args = command(tmp_path)
    entry, _ = ready(tmp_path, monkeypatch, args)
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        assert path.suffix != ".pch", "PCH payload read during metadata validation"
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    assert common.read_manifest(entry, common.compile_flags(args), GENERATION)["files"]


def test_warm_command_retains_module_macro_and_publishes_only_complete_entry(tmp_path, monkeypatch):
    args = command(tmp_path)
    flags = common.compile_flags(args)
    warm_flags = common.strip_outputs(args, drop_module=False)
    monkeypatch.setenv("KERNELBOT_NVCC_PCH_ROOT", str(tmp_path / "warm"))
    monkeypatch.setattr(warmer, "capture", lambda *args: (flags, warm_flags))

    def compile_header(argv, **kwargs):
        assert "-DTORCH_EXTENSION_NAME=first" in argv
        pch_dir = Path(argv[argv.index("--pch-dir") + 1])
        assert not (pch_dir.parent / "manifest.json").exists()
        (pch_dir / "cicc_ab_12.pch").write_bytes(b"device")
        (pch_dir / "cudafe_ab_12.pch").write_bytes(b"frontend")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(warmer.subprocess, "run", compile_header)
    result = warmer.warm_profile("sm75-default", GENERATION)
    entry = common.cache_root() / GENERATION / result["key"]
    assert common.read_manifest(entry, flags, GENERATION) == result
    assert not (entry / "manifest.json.tmp").exists()
