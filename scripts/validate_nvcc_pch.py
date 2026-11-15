"""Validate the real CPU-to-GPU submission path without publishing leaderboard scores.

KERNELBOT_NVCC_TOOLCHAIN=1 KERNELBOT_NVCC_PCH=1 \
  PYTHONPATH=src:src/runners uv run modal run scripts/validate_nvcc_pch.py

Publish the verified private toolchain Volume with upload_nvcc_toolchain.py first.
The default matrix runs all edge cases on T4 and PCH-on/off on H100 and B200.
The two baseline cases execute the normal test, benchmark, and leaderboard
validation phases in an ephemeral app; no service or database is contacted.
Use --profiles with trusted nvcc_common.PROFILES names to select baseline
flag sets, and --repeats to repeat baseline pairs in fresh containers.
The optional CUTLASS pair uses the generic sm75-O3 framework profile:
  --gpus T4 --cases cutlass,cutlass-disabled --repeats 2 --concurrency 1
"""

import ast
import asyncio
import dataclasses
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

os.environ["KERNELBOT_MODAL_APP"] = os.environ.get(
    "KERNELBOT_NVCC_VALIDATION_APP", "kernelbot-nvcc-pch-release-validation"
)
if os.environ["KERNELBOT_MODAL_APP"] == "discord-bot-runner":
    raise RuntimeError("The validation app must be separate from the production runner")
os.environ.setdefault("KERNELBOT_NVCC_TOOLCHAIN", "1")
os.environ.setdefault("KERNELBOT_NVCC_PCH", "1")
os.environ.setdefault("KERNELBOT_PCH_VOLUME", "kernelbot-nvcc-release-host-pch")
os.environ.setdefault("KERNELBOT_NVCC_PCH_VOLUME", "kernelbot-nvcc-release-device-pch")

from modal_runner import (  # noqa: E402
    NVCC_PCH_MOUNT,
    app,
    cuda_image,
    submission_volumes,
    warm_nvcc_pch,
    warm_pch,
)

from libkernelbot.consts import GPU_TO_SM, SubmissionMode, Timeout  # noqa: E402
from libkernelbot.python_precompile import (  # noqa: E402
    compile_python,
    run_with_cpu_artifacts,
)
from libkernelbot.run_eval import run_config  # noqa: E402
from libkernelbot.task import build_task_config, make_task_definition  # noqa: E402

# Serialized GPU functions refer to helpers from this module by name. Mount the
# same harness source on both stages so fresh containers can import those helpers.
cuda_image = cuda_image.add_local_file(Path(__file__).resolve(), "/root/validate_nvcc_pch.py")

GPUS = ("T4", "L4", "A100", "H100", "B200")
DEFAULT_GPUS = ("T4", "H100", "B200")
GPU_RESOURCES = {"A100": "A100-80GB", "H100": "H100!"}
BASELINE_CASES = ("default", "disabled")
CUTLASS_CASES = ("cutlass", "cutlass-disabled")
CASES = (
    *BASELINE_CASES,
    "no-implicit",
    "unknown-flags",
    "changed-header",
    "rejected-cache",
    "empty-cache",
    "custom-nvcc",
)
ALL_CASES = (*CASES, *CUTLASS_CASES)
FINGERPRINT = Path("/opt/kernelbot-pch/nvcc-fingerprint")
GPU_ARCHES = {
    "T4": {"75"},
    "L4": {"89"},
    "A100": {"80"},
    "H100": {"90", "90a"},
    "B200": {"100", "100a"},
}


def profile_catalog() -> dict:
    # Only the local orchestrator needs the profile catalog. The remote image
    # keeps these stdlib modules in /opt/kernelbot-pch instead of a Python package.
    from pch.nvcc_common import PROFILES

    return PROFILES


def make_submission(example: Path, case: str, extra_cuda_cflags=()) -> str:
    source = (example / "submission_cuda_inline.py").read_text()
    source = source.replace("name='add_cuda'", "name='nvcc_release_vectoradd'")
    if extra_cuda_cflags:
        source = source.replace(
            "    verbose=True,",
            f"    verbose=True,\n    extra_cuda_cflags={list(extra_cuda_cflags)!r},",
        )
    if case == "no-implicit":
        source = source.replace(
            'add_cuda_source = """',
            'add_cuda_source = """\n#include <torch/types.h>\n#include <cuda_runtime.h>',
        ).replace("    verbose=True,", "    verbose=True,\n    no_implicit_headers=True,")
    elif case == "unknown-flags":
        source = source.replace(
            "    verbose=True,",
            "    verbose=True,\n    extra_cuda_cflags=['-DKERNELBOT_RELEASE_PROFILE_MISS=1'],",
        )
    elif case == "changed-header":
        # Identical source and options in both containers, but a genuinely changed
        # user dependency. Replaying the CPU binary would produce wrong answers.
        source = (
            "import torch\nfrom pathlib import Path\n"
            "Path('/tmp/kernelbot-nvcc-release-user.h').write_text(\n"
            "    '#define OFFSET ' + ('0' if torch.cuda.is_available() else '1') + '\\n')\n"
            + source.replace(
                'add_cuda_source = """',
                'add_cuda_source = """\n#include "/tmp/kernelbot-nvcc-release-user.h"',
            ).replace("C[idx] = A[idx] + B[idx];", "C[idx] = A[idx] + B[idx] + OFFSET;")
        )
    elif case == "custom-nvcc":
        source = "import os\nos.environ['PYTORCH_NVCC'] = '/usr/local/cuda/bin/nvcc'\n" + source
    return source


def cutlass_submission() -> str:
    # Reuse the benchmark's actual SM75 tensor-core GEMM without importing its
    # Modal app or its workload-specific PCH. Only these two string literals run.
    benchmark = Path(__file__).with_name("modal_pch_benchmark.py")
    constants = {}
    for node in ast.parse(benchmark.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in {"GEMM_CPP", "GEMM_CUDA"}:
                    constants[target.id] = (
                        ast.literal_eval(node.value)
                        .replace('#include "variant.h"', "constexpr int kVariant = 1;")
                        .replace("TILE_N", "128")
                    )
    return (
        "# GEMM fixture adapted from scripts/modal_pch_benchmark.py.\n"
        "import torch\nfrom torch.utils.cpp_extension import load_inline\n"
        "module = load_inline(name='nvcc_release_cutlass',\n"
        f"    cpp_sources={constants['GEMM_CPP']!r},\n"
        f"    cuda_sources={constants['GEMM_CUDA']!r},\n"
        "    functions=['add_cuda', 'version'], extra_cuda_cflags=['-O3'],\n"
        "    no_implicit_headers=False, verbose=True)\n"
        "assert module.version() == 1\n"
        "def custom_kernel(data):\n"
        "    a, b, nondefault = data\n"
        "    if not nondefault:\n"
        "        return module.add_cuda(a, b)\n"
        "    current = torch.cuda.current_stream(a.device)\n"
        "    stream = torch.cuda.Stream(device=a.device)\n"
        "    stream.wait_stream(current)\n"
        "    with torch.cuda.stream(stream):\n"
        "        result = module.add_cuda(a, b)\n"
        "        a.record_stream(stream)\n"
        "        b.record_stream(stream)\n"
        "    current.wait_stream(stream)\n"
        "    result.record_stream(current)\n"
        "    return result\n"
    )


CUTLASS_REFERENCE = """
import torch

def generate_input(m, n, k, seed, zero=0, ones=0, nondefault=1):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    a = torch.randn(m, k, device='cuda', dtype=torch.float16, generator=generator) / 4
    b = torch.randn(n, k, device='cuda', dtype=torch.float16, generator=generator) / 4
    if zero:
        a.zero_()
    if ones:
        a.fill_(1)
        b.fill_(1)
    return a, b, nondefault

def check_implementation(data, output):
    a, b, _ = data
    expected = a.double() @ b.double().T
    try:
        torch.testing.assert_close(output.double(), expected, rtol=2e-4, atol=2e-4)
    except AssertionError as error:
        return False, str(error)
    error = float((output.double() - expected).abs().max())
    return True, f'max_absolute_error={error}'
"""


def configure_cutlass(config):
    config["sources"]["reference.py"] = CUTLASS_REFERENCE
    config["sources"]["task.py"] = "TestSpec = dict\n"
    config["tests"] = [
        {"m": m, "n": n, "k": k, "seed": 17, "zero": zero, "nondefault": 1}
        for m, n, k in ((128, 128, 64), (129, 136, 64), (256, 256, 128), (512, 512, 256))
        for zero in (0, 1)
    ] + [{"m": 8, "n": 8, "k": 8, "seed": 17, "ones": 1, "nondefault": 1}]
    # Time the kernel on the usual stream; correctness tests above exercise a
    # separate stream, random/zero inputs, and tile tails against float64 Torch.
    config["benchmarks"] = [{"m": 256, "n": 256, "k": 128, "seed": 54352, "nondefault": 0}]


def make_config(gpu: str, case: str, profile: str | None = None) -> dict:
    example = Path(__file__).resolve().parents[1] / "examples/vectoradd_py"
    if case in CUTLASS_CASES:
        profile = profile or "sm75-O3"
        if gpu != "T4" or profile != "sm75-O3":
            raise ValueError("The CUTLASS fixture requires T4 and the generic sm75-O3 profile")
    arch, flags = profile_catalog()[profile or f"sm{GPU_TO_SM[gpu]}-default"]
    if arch not in GPU_ARCHES[gpu]:
        raise ValueError(f"Profile {profile} targets SM{arch}, which is incompatible with {gpu}")
    task = make_task_definition(example).task
    mode = (
        SubmissionMode.LEADERBOARD
        if case in (*BASELINE_CASES, *CUTLASS_CASES)
        else SubmissionMode.TEST
    )
    config = build_task_config(
        task=task,
        submission_content=(
            cutlass_submission() if case in CUTLASS_CASES else make_submission(example, case, flags)
        ),
        arch=arch,
        mode=mode,
    )
    config["benchmarks"] = [{"size": 512, "seed": 54352}]
    if case in CUTLASS_CASES:
        configure_cutlass(config)
    return config


def resource_usage() -> dict:
    import resource

    result = {
        "self_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "child_peak_rss_kib": resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
    }
    for name in ("memory.peak", "memory.max"):
        path = Path("/sys/fs/cgroup") / name
        if path.exists():
            result[name] = path.read_text().strip()
    return result


def read_events(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def prepare_rejected_cache(destination: Path, arch: str) -> dict:
    """Damage disposable sparse copies; never modify files on the shared Volume."""
    generation = FINGERPRINT.read_text().strip()
    manifests = list((Path(NVCC_PCH_MOUNT) / generation).glob("*/manifest.json"))
    for path in manifests:
        manifest = json.loads(path.read_text())
        if manifest.get("profile") != f"sm{arch}-default":
            continue
        entry = destination / generation / path.parent.name
        (entry / "pch").mkdir(parents=True)
        shutil.copyfile(path, entry / "manifest.json")
        shutil.copyfile(path.parent / "gpumode.cuh", entry / "gpumode.cuh")
        total = 0
        for item in manifest["files"]:
            target = entry / item["path"]
            if not target.resolve().is_relative_to((entry / "pch").resolve()):
                raise ValueError("Unexpected PCH path in trusted manifest")
            target.parent.mkdir(exist_ok=True)
            with target.open("wb") as handle:
                # Preserve manifest size validation while making the PCH invalid.
                # Sparse local files do not copy the multi-GB shared payload.
                handle.truncate(item["bytes"])
            total += item["bytes"]
        return {"profile": manifest["profile"], "sparse_bytes": total}
    raise RuntimeError(f"Warm sm{arch}-default before running the rejection test")


@contextmanager
def validation_environment(case: str, arch: str):
    previous = dict(os.environ)
    event_path = Path("/tmp/kernelbot-nvcc-release-events.jsonl")
    event_path.unlink(missing_ok=True)
    metadata = {}
    try:
        os.environ.update(
            KERNELBOT_NVCC_PCH="1",
            KERNELBOT_NVCC_PCH_DISABLE="1" if case in {"disabled", "cutlass-disabled"} else "0",
            KERNELBOT_NVCC_PCH_EVENTS=str(event_path),
        )
        os.environ.pop("KERNELBOT_NVCC_PCH_ROOT", None)
        if case in {"rejected-cache", "empty-cache"}:
            destination = Path("/tmp") / f"kernelbot-nvcc-release-{case}"
            destination.mkdir(exist_ok=False)
            if case == "rejected-cache":
                metadata = prepare_rejected_cache(destination, arch)
            os.environ["KERNELBOT_NVCC_PCH_ROOT"] = str(destination)
        yield event_path, metadata
    finally:
        os.environ.clear()
        os.environ.update(previous)


def toolchain_info() -> dict:
    return {
        "generation": FINGERPRINT.read_text().strip(),
        "nvcc_version": subprocess.check_output(
            ["/usr/local/cuda/bin/nvcc", "--version"], text=True
        ),
        "container_id": os.environ["MODAL_TASK_ID"],
        "image_id": os.environ["MODAL_IMAGE_ID"],
    }


@app.function(
    image=cuda_image,
    cpu=4,
    memory=8192,
    timeout=Timeout.COMPILE + 30,
    max_containers=3,
    single_use_containers=True,
    restrict_modal_access=True,
    volumes=submission_volumes,
)
def validate_cpu(config: dict, case: str) -> dict:
    started = time.perf_counter()
    with validation_environment(case, config["arch"]) as (event_path, metadata):
        packet = compile_python(config)
        return {
            "packet": packet,
            "events": read_events(event_path),
            "toolchain": toolchain_info(),
            "memory": resource_usage(),
            "case_metadata": metadata,
            "wall_seconds": time.perf_counter() - started,
            "requested_memory_mib": 8192,
            "max_jobs": 2,
        }


def validate_gpu(config: dict, case: str, packet: dict) -> dict:
    started = time.perf_counter()
    with validation_environment(case, config["arch"]) as (event_path, metadata):
        result = run_with_cpu_artifacts({**config, "cpu_compile": packet}, run_config)
        return {
            "result": dataclasses.asdict(result),
            "events": read_events(event_path),
            "toolchain": toolchain_info(),
            "memory": resource_usage(),
            "case_metadata": metadata,
            "wall_seconds": time.perf_counter() - started,
        }


gpu_functions = {
    gpu: app.function(
        image=cuda_image,
        gpu=GPU_RESOURCES.get(gpu, gpu),
        cpu=4,
        memory=16384,
        timeout=600,
        max_containers=1,
        single_use_containers=True,
        restrict_modal_access=True,
        volumes=submission_volumes,
        name=f"validate_nvcc_{gpu.lower()}",
        serialized=True,
    )(validate_gpu)
    for gpu in GPUS
}


def verify_hits(events, profile):
    for event in events:
        if event["status"] == "hit":
            assert event["frontends"] == ["cicc", "cudafe"], event
            if profile:
                assert event["profile"] == profile, event


def verify_result(case: str, cpu: dict, gpu: dict, profile: str | None = None) -> None:
    packet = cpu["packet"]
    assert packet["info"]["status"] == "compiled", packet["info"]
    assert packet["info"]["artifacts"] > 0 and packet["artifacts"]
    assert cpu["toolchain"]["generation"] == gpu["toolchain"]["generation"]
    assert cpu["toolchain"]["image_id"] == gpu["toolchain"]["image_id"]
    result = gpu["result"]
    assert result["success"], result["error"]
    assert result["runs"], "No evaluation phases ran"
    for name, evaluation in result["runs"].items():
        run = evaluation["run"]
        assert run and run["success"] and run["passed"], (name, evaluation)
        assert run["result"].get("check") == "pass", (name, run["result"])
    info = result["cpu_compile"]
    cpu_statuses = {event["status"] for event in cpu["events"]}
    gpu_statuses = {event["status"] for event in gpu["events"]}
    if case == "changed-header":
        assert info["fallbacks"] > 0 and info["reused"] == 0, info
        assert "dependency changed" in info["reason"].lower(), info
        assert "hit" in gpu_statuses, gpu["events"]
    else:
        assert info["status"] == "reused" and info["reused"] >= 2, info
        assert info["fallbacks"] == 0, info
        assert not gpu["events"], gpu["events"]
    expected_status = {
        "default": "hit",
        "disabled": "disabled",
        "no-implicit": "bypass",
        "unknown-flags": "miss",
        "changed-header": "hit",
        "rejected-cache": "fallback",
        "empty-cache": "miss",
        "cutlass": "hit",
        "cutlass-disabled": "disabled",
    }
    if case == "custom-nvcc":
        assert not cpu["events"], cpu["events"]
    else:
        assert cpu_statuses == {expected_status[case]}, cpu["events"]
    verify_hits(cpu["events"] + gpu["events"], profile)
    if case not in {"default", "changed-header", "cutlass"}:
        assert "hit" not in cpu_statuses, cpu["events"]
    if case == "rejected-cache":
        assert any(
            event["status"] == "fallback" and event["reason"] == "cache_rejected"
            for event in cpu["events"]
        ), cpu["events"]
    if case in (*BASELINE_CASES, *CUTLASS_CASES):
        assert set(result["runs"]) == {"test", "benchmark", "leaderboard"}
        measurements = result["runs"]["benchmark"]["run"]["result"]
        assert float(measurements["benchmark.0.mean"]) > 0, measurements


def compact_cpu(cpu: dict) -> dict:
    packet = cpu["packet"]
    binary = packet["artifacts"]
    return cpu | {
        "packet": {key: value for key, value in packet.items() if key != "artifacts"}
        | {"artifact_bytes": len(binary), "artifact_sha256": hashlib.sha256(binary).hexdigest()}
    }


def paired_jobs(gpu, profile, pair, cases, repeats):
    return [
        (gpu, case, profile, repeat)
        for repeat in range(1, repeats + 1)
        for case in pair[:: 1 if repeat % 2 else -1]
        if case in cases
    ]


def matrix_jobs(gpus, cases, edge_gpu, profiles, repeats):
    catalog = profile_catalog()
    selected = profiles or [f"sm{GPU_TO_SM[gpu]}-default" for gpu in gpus]
    if len(selected) > 12 or not set(selected) <= catalog.keys():
        raise ValueError("Choose at most 12 fixed profiles from " + ", ".join(catalog))
    jobs = []
    for profile in selected:
        arch, _ = catalog[profile]
        matching = [gpu for gpu in gpus if arch in GPU_ARCHES[gpu]]
        if len(matching) != 1:
            raise ValueError(f"Select a compatible GPU for {profile} (SM{arch})")
        jobs.extend(paired_jobs(matching[0], profile, BASELINE_CASES, cases, repeats))
    if set(cases) & set(CUTLASS_CASES):
        if "T4" not in gpus:
            raise ValueError("Select T4 for the CUTLASS fixture")
        jobs.extend(paired_jobs("T4", "sm75-O3", CUTLASS_CASES, cases, repeats))
    if edge_gpu in gpus:
        jobs.extend(
            (edge_gpu, case, f"sm{GPU_TO_SM[edge_gpu]}-default", 1)
            for case in cases
            if case not in (*BASELINE_CASES, *CUTLASS_CASES)
        )
    if not jobs:
        raise ValueError("No jobs selected; edge cases run only on edge_gpu")
    return jobs


def comparisons(rows):
    """Keep latency observations separate from the functional pass/fail gates."""
    grouped = {}
    failed = set()
    for row in rows:
        if row["case"] in (*BASELINE_CASES, *CUTLASS_CASES):
            workload = "cutlass" if row["case"] in CUTLASS_CASES else "vectoradd"
            arm = "disabled" if row["case"].endswith("disabled") else "default"
            key = (row["gpu"], row["profile"], workload)
            if not row["passed"]:
                failed.add(key)
            else:
                grouped.setdefault(key, {}).setdefault(arm, []).append(
                    row["cpu"]["packet"]["info"]["duration"]
                )
    results = []
    for (gpu, profile, workload), arms in sorted(grouped.items()):
        if (
            (gpu, profile, workload) in failed
            or set(arms) != set(BASELINE_CASES)
            or len(arms["default"]) != len(arms["disabled"])
        ):
            continue
        on, off = (statistics.median(arms[case]) for case in BASELINE_CASES)
        results.append(
            {
                "gpu": gpu,
                "profile": profile,
                "workload": workload,
                "cpu_compile_seconds": arms,
                "median_pch_on_seconds": on,
                "median_pch_off_seconds": off,
                "speedup": off / on,
                "reduction_percent": 100 * (1 - on / off),
                "scope": "CPU compile_python including import, dependency validation and packing",
            }
        )
    return results


@app.local_entrypoint()
async def main(
    gpus: str = ",".join(DEFAULT_GPUS),
    cases: str = ",".join(CASES),
    edge_gpu: str = "T4",
    profiles: str = "",
    repeats: int = 1,
    skip_warm: bool = False,
    concurrency: int = 2,
    output: str = "/tmp/kernelbot-nvcc-pch-release-validation.json",
):
    selected_gpus, selected_cases = gpus.split(","), cases.split(",")
    if not set(selected_gpus) <= set(GPUS) or not set(selected_cases) <= set(ALL_CASES):
        raise ValueError(f"Choose GPUs from {GPUS} and cases from {ALL_CASES}")
    if edge_gpu not in GPUS or not 1 <= concurrency <= 3 or not 1 <= repeats <= 5:
        raise ValueError("Choose a supported edge_gpu, concurrency 1–3, and repeats 1–5")
    selected_profiles = list(dict.fromkeys(profiles.split(","))) if profiles else []
    jobs = matrix_jobs(selected_gpus, selected_cases, edge_gpu, selected_profiles, repeats)
    result = {
        "app": os.environ["KERNELBOT_MODAL_APP"],
        "baseline_repeats": repeats,
        "concurrency": concurrency,
        "rows": [],
    }

    def save():
        result["comparisons"] = comparisons(result["rows"])
        Path(output).write_text(json.dumps(result, default=str, indent=2) + "\n")

    if not skip_warm:
        result["host_warmup"] = await warm_pch.remote.aio(profiles="cuda-default")
        warm_profiles = ",".join(dict.fromkeys(job[2] for job in jobs))
        result["nvcc_warmup"] = await warm_nvcc_pch.remote.aio(profiles=warm_profiles)
        save()
    gate = asyncio.Semaphore(concurrency)

    async def run_one(gpu_name, case, profile, repeat):
        async with gate:
            arch, flags = profile_catalog()[profile]
            row = {
                "gpu": gpu_name,
                "case": case,
                "workload": "cutlass" if case in CUTLASS_CASES else "vectoradd",
                "profile": profile,
                "arch": arch,
                "extra_cuda_cflags": flags,
                "repeat": repeat,
                "passed": False,
                "state": "running",
            }
            started = time.perf_counter()
            result["rows"].append(row)
            save()
            try:
                config = make_config(gpu_name, case, profile)
                row["sources_sha256"] = hashlib.sha256(
                    json.dumps(config["sources"], sort_keys=True).encode()
                ).hexdigest()
                cpu = await validate_cpu.remote.aio(config=config, case=case)
                row["cpu"] = compact_cpu(cpu)
                save()
                gpu = await gpu_functions[gpu_name].remote.aio(
                    config=config, case=case, packet=cpu["packet"]
                )
                row["gpu_run"] = gpu
                verify_result(case, cpu, gpu, profile)
                row["passed"] = True
            except Exception as exc:
                row["error"] = f"{type(exc).__name__}: {exc}"
            row["wall_seconds"] = time.perf_counter() - started
            row["state"] = "passed" if row["passed"] else "failed"
            save()
            print(
                json.dumps(
                    {
                        key: row.get(key)
                        for key in ("gpu", "case", "profile", "repeat", "passed", "error")
                    }
                ),
                flush=True,
            )

    await asyncio.gather(*(run_one(*job) for job in jobs))
    identifiers = [
        stage["toolchain"]["container_id"]
        for row in result["rows"]
        for stage in (row.get("cpu"), row.get("gpu_run"))
        if stage
    ]
    assert len(identifiers) == len(set(identifiers)), "Expected fresh CPU and GPU containers"
    assert all(row["passed"] for row in result["rows"]), f"Validation failures saved in {output}"
    print(f"Passed {len(result['rows'])} CPU-to-GPU cases; results: {output}", flush=True)
