"""Exercise release-matrix selection and pass gates without running Modal jobs."""

import ast
import copy
import os
import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


@pytest.fixture(scope="module")
def harness():
    root = Path(__file__).parents[1]
    with pytest.MonkeyPatch.context() as monkeypatch, patch.dict(os.environ):
        monkeypatch.syspath_prepend(str(root / "src/runners"))
        namespace = runpy.run_path(str(root / "scripts/validate_nvcc_pch.py"))
        namespace["profile_catalog"]()
    return SimpleNamespace(**namespace)


def test_default_matrix_preserves_twelve_fresh_cpu_gpu_cases(harness):
    jobs = harness.matrix_jobs(harness.DEFAULT_GPUS, harness.CASES, "T4", [], 1)
    assert len(jobs) == 12
    assert {case for gpu, case, _, _ in jobs if gpu == "T4"} == set(harness.CASES)
    assert {case for gpu, case, _, _ in jobs if gpu != "T4"} == set(harness.BASELINE_CASES)


def test_all_gpu_matrix_adds_l4_and_production_a100_resource(harness):
    jobs = harness.matrix_jobs(harness.GPUS, harness.CASES, "T4", [], 1)
    assert len(jobs) == 16
    assert harness.GPU_RESOURCES["A100"] == "A100-80GB"
    assert harness.GPU_RESOURCES.get("L4", "L4") == "L4"
    assert {profile for gpu, _, profile, _ in jobs if gpu == "A100"} == {"sm80-default"}
    assert {profile for gpu, _, profile, _ in jobs if gpu == "L4"} == {"sm89-default"}
    assert harness.make_config("A100", "default")["arch"] == "80"
    assert harness.make_config("L4", "default")["arch"] == "89"


def test_observed_sm80_profile_runs_only_on_a100(harness):
    profile = "sm80-O3-cxx17-arch-fast-math"
    jobs = harness.matrix_jobs(harness.GPUS, harness.BASELINE_CASES, "T4", [profile], 1)
    assert jobs == [
        ("A100", "default", profile, 1),
        ("A100", "disabled", profile, 1),
    ]
    assert harness.make_config("A100", "default", profile)["arch"] == "80"
    with pytest.raises(ValueError, match="incompatible"):
        harness.make_config("L4", "default", profile)


def test_selected_profiles_repeat_paired_baselines_and_alternate_order(harness):
    jobs = harness.matrix_jobs(["B200"], harness.BASELINE_CASES, "T4", ["sm100-O3"], 2)
    assert jobs == [
        ("B200", "default", "sm100-O3", 1),
        ("B200", "disabled", "sm100-O3", 1),
        ("B200", "disabled", "sm100-O3", 2),
        ("B200", "default", "sm100-O3", 2),
    ]


def test_cutlass_pair_uses_generic_o3_profile_without_expanding_default_matrix(harness):
    jobs = harness.matrix_jobs(["T4"], harness.CUTLASS_CASES, "T4", [], 2)
    assert jobs == [
        ("T4", "cutlass", "sm75-O3", 1),
        ("T4", "cutlass-disabled", "sm75-O3", 1),
        ("T4", "cutlass-disabled", "sm75-O3", 2),
        ("T4", "cutlass", "sm75-O3", 2),
    ]
    assert not set(harness.CASES) & set(harness.CUTLASS_CASES)
    with pytest.raises(ValueError, match="Select T4"):
        harness.matrix_jobs(["B200"], harness.CUTLASS_CASES, "T4", [], 1)


def test_cutlass_fixture_has_real_gemm_and_tail_zero_stream_checks(harness):
    on = harness.make_config("T4", "cutlass")
    off = harness.make_config("T4", "cutlass-disabled")
    assert on == off
    assert on["arch"] == "75"
    source = on["sources"]["submission.py"]
    assert "cutlass::gemm::device::Gemm" in source
    assert "c10::cuda::getCurrentCUDAStream" in source
    assert "torch.cuda.stream(stream)" in source
    tree = ast.parse(source)
    call = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_inline"
    )
    kwargs = {keyword.arg: keyword.value for keyword in call.keywords}
    assert ast.literal_eval(kwargs["extra_cuda_cflags"]) == ["-O3"]
    assert ast.literal_eval(kwargs["no_implicit_headers"]) is False
    assert len(on["tests"]) == 9
    assert any(test["m"] == 129 and test["n"] == 136 for test in on["tests"])
    assert sum(test.get("zero", 0) for test in on["tests"]) == 4
    assert all(test["nondefault"] == 1 for test in on["tests"])
    assert on["benchmarks"][0]["nondefault"] == 0
    assert "a.double() @ b.double().T" in on["sources"]["reference.py"]
    for name, contents in on["sources"].items():
        ast.parse(contents, filename=name)


def test_explicit_arch_profile_sets_architecture_and_exact_extra_flags(harness, monkeypatch):
    profile, flags = "sm100a-test", ["-O3", "--use_fast_math", "-arch=sm_100a"]
    catalog = {profile: ("100a", flags)}
    monkeypatch.setitem(harness.make_config.__globals__, "profile_catalog", lambda: catalog)
    jobs = harness.matrix_jobs(["B200"], harness.BASELINE_CASES, "T4", [profile], 1)
    assert {job[0] for job in jobs} == {"B200"}
    on = harness.make_config("B200", "default", profile)
    off = harness.make_config("B200", "disabled", profile)
    assert on == off
    assert on["arch"] == "100a"
    tree = ast.parse(on["sources"]["submission.py"])
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "load_inline"
    ]
    kwargs = {keyword.arg: keyword.value for keyword in calls[0].keywords}
    assert ast.literal_eval(kwargs["extra_cuda_cflags"]) == flags


@pytest.mark.parametrize("profile", ["sm100-default", "-O3; arbitrary-command"])
def test_profile_selection_rejects_unknown_or_incompatible_architecture(harness, profile):
    with pytest.raises(ValueError):
        harness.matrix_jobs(["T4"], harness.BASELINE_CASES, "T4", [profile], 1)


def test_comparison_requires_paired_successful_observations(harness):
    def row(case, seconds, passed=True):
        return {
            "gpu": "T4",
            "profile": "sm75-default",
            "case": case,
            "passed": passed,
            "cpu": {"packet": {"info": {"duration": seconds}}},
        }

    rows = [row("default", 10), row("default", 20), row("disabled", 25), row("disabled", 35)]
    result = harness.comparisons(rows)[0]
    assert result["median_pch_on_seconds"] == 15
    assert result["median_pch_off_seconds"] == 30
    assert result["speedup"] == 2
    assert result["reduction_percent"] == 50
    assert result["cpu_compile_seconds"]["default"] == [10, 20]
    assert harness.comparisons(rows[:-1]) == []
    assert harness.comparisons(rows + [row("default", 100, False)]) == []

    # A vectoradd profile run and the CUTLASS fixture must remain separate even
    # when both use the same generic framework-header profile and GPU.
    cutlass = [row("cutlass", 40), row("cutlass-disabled", 50)]
    results = harness.comparisons(rows + cutlass)
    assert {item["workload"]: item["speedup"] for item in results} == {
        "vectoradd": 2,
        "cutlass": 1.25,
    }


@pytest.fixture
def successful_pair():
    toolchain = {"generation": "same-generation", "image_id": "same-image"}
    event = {"status": "hit", "frontends": ["cicc", "cudafe"], "profile": "sm75-default"}
    run = {"success": True, "passed": True, "result": {"check": "pass", "benchmark.0.mean": 1}}
    return (
        {
            "packet": {"info": {"status": "compiled", "artifacts": 1}, "artifacts": b"artifact"},
            "toolchain": toolchain,
            "events": [event],
        },
        {
            "toolchain": toolchain,
            "events": [],
            "result": {
                "success": True,
                "error": "",
                "runs": {name: {"run": run} for name in ("test", "benchmark", "leaderboard")},
                "cpu_compile": {"status": "reused", "reused": 6, "fallbacks": 0},
            },
        },
    )


def test_success_requires_matching_profile_and_no_gpu_compilation(harness, successful_pair):
    cpu, gpu = successful_pair
    harness.verify_result("default", cpu, gpu, "sm75-default")
    with pytest.raises(AssertionError):
        harness.verify_result("default", cpu, gpu, "sm75-O3")
    gpu["events"] = [{"status": "bypass"}]
    with pytest.raises(AssertionError):
        harness.verify_result("default", cpu, gpu, "sm75-default")


def test_changed_header_requires_rejection_and_gpu_pch_hit(harness, successful_pair):
    cpu, gpu = copy.deepcopy(successful_pair)
    gpu["events"] = cpu["events"]
    info = {
        "status": "fallback",
        "reused": 0,
        "fallbacks": 2,
        "reason": "Build dependency changed: /tmp/user.h",
    }
    gpu["result"]["cpu_compile"] = info
    harness.verify_result("changed-header", cpu, gpu, "sm75-default")
    info["reused"] = 1
    with pytest.raises(AssertionError):
        harness.verify_result("changed-header", cpu, gpu, "sm75-default")
