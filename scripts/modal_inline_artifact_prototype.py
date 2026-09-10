"""Run the CPU-build experiment described in docs/modal-inline-artifact-prototype.md."""

import dataclasses
import json
import os
import subprocess
import tempfile
import time
from pathlib import Path

import modal
from modal_runner import cuda_image

from libkernelbot.inline_artifacts import (
    pack_artifacts,
    prepare_environment,
    read_events,
    unpack_artifacts,
)

app = modal.App("kernelbot-inline-artifact-prototype")
GPU = os.environ.get("KERNELBOT_PROTOTYPE_GPU", "T4").upper()
ARCH = {"T4": "7.5", "L4": "8.9", "A100": "8.0", "H100": "9.0a", "B200": "10.0a"}[GPU]


def _write_sources(work: Path, sources: dict[str, str]) -> None:
    for name, source in sources.items():
        path = work / name
        if not path.resolve().is_relative_to(work.resolve()):
            raise ValueError(f"Invalid source path: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)


def _import_submission(work: Path, environment: dict) -> dict:
    started = time.perf_counter()
    result = subprocess.run(
        ["python3", "submission.py"],
        cwd=work,
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
    )
    if result.returncode:
        raise RuntimeError(f"Submission import failed:\n{result.stdout}\n{result.stderr}")
    return {
        "process_seconds": time.perf_counter() - started,
        "events": read_events(work / "artifacts"),
    }


@app.function(
    image=cuda_image, cpu=4, memory=8192, timeout=300, max_containers=1, scaledown_window=2
)
def compile_cpu(sources: dict[str, str], arch: str) -> dict:
    import torch

    if torch.cuda.is_available() or torch.cuda.device_count() != 0:
        raise RuntimeError("CPU build stage unexpectedly has a GPU")
    with tempfile.TemporaryDirectory(prefix="kb-cpu-build-") as directory:
        work = Path(directory)
        _write_sources(work, sources)
        result = _import_submission(work, prepare_environment(work, "capture", arch))
        artifacts = pack_artifacts(work / "artifacts")
        if not artifacts:
            raise RuntimeError(
                "No import-time load_inline calls captured; "
                "lazy/GPU-dependent builds are unsupported"
            )
        result.update({"cuda_available": False, "device_count": 0, "artifacts": artifacts})
        return result


@app.function(
    image=cuda_image, gpu=GPU, cpu=4, memory=8192, timeout=600, max_containers=1, scaledown_window=2
)
def evaluate_gpu(
    sources: dict[str, str], artifacts: dict[str, bytes], arch: str, tests: str
) -> dict:
    import torch

    from libkernelbot.run_eval import make_system_info, run_single_evaluation

    expected_capability = tuple(int(x) for x in arch.rstrip("a").split("."))
    if torch.cuda.get_device_capability() != expected_capability:
        raise RuntimeError("GPU architecture does not match the CPU compilation target")
    started = time.perf_counter()
    result = {"gpu": torch.cuda.get_device_name(), "capability": expected_capability}
    with tempfile.TemporaryDirectory(prefix="kb-gpu-replay-") as directory:
        work = Path(directory)
        _write_sources(work, sources)
        unpack_artifacts(work / "artifacts", artifacts)
        environment = prepare_environment(work, "replay", arch)
        result["replay_import"] = _import_submission(work, environment)

        # Keep KernelBot's normal test protocol, including its multiprocessing evaluator.
        previous_cwd, previous_environment = Path.cwd(), dict(os.environ)
        try:
            os.chdir(work)
            os.environ.update(environment)
            run, _ = run_single_evaluation(
                ["python3", "eval.py"],
                "test",
                system=make_system_info(),
                tests=tests,
                seed=42,
            )
        finally:
            os.chdir(previous_cwd)
            os.environ.clear()
            os.environ.update(previous_environment)
        result["evaluation"] = dataclasses.asdict(run)
        if not run.success or not run.passed:
            raise RuntimeError(f"Artifact evaluation failed: {result['evaluation']}")
        result["replay_events"] = read_events(work / "artifacts")
        if not result["replay_events"] or any(
            e["mode"] != "replay" for e in result["replay_events"]
        ):
            raise RuntimeError("Expected artifact-only GPU execution")
        if (work / "build").exists():
            raise RuntimeError("GPU replay unexpectedly created a build directory")

        # Negative control: missing binaries must fail instead of silently recompiling.
        missing = work / "missing"
        missing.mkdir()
        _write_sources(missing, sources)
        probe = subprocess.run(
            ["python3", "submission.py"],
            cwd=missing,
            env=prepare_environment(missing, "replay", arch),
            capture_output=True,
            text=True,
            timeout=60,
        )
        result["missing_artifact_rejected"] = (
            probe.returncode != 0 and "No CPU-built artifact" in probe.stderr
        )
        if not result["missing_artifact_rejected"]:
            raise RuntimeError(f"Missing-artifact negative control failed: {probe.stderr}")

        # Compare against a fresh GPU-side build using the same sources and image.
        baseline = work / "baseline"
        baseline.mkdir()
        _write_sources(baseline, sources)
        result["gpu_compile_baseline"] = _import_submission(
            baseline, prepare_environment(baseline, "capture", arch)
        )
    result["gpu_function_seconds"] = time.perf_counter() - started
    return result


@app.local_entrypoint()
def main(output: str = "", submission: str = ""):
    repo = Path(__file__).resolve().parents[1]
    example = repo / "examples" / "vectoradd_py"
    sources = {
        "submission.py": Path(submission).read_text()
        if submission
        else (example / "submission_cuda_inline.py").read_text(),
        "task.py": (example / "task.py").read_text(),
        "reference.py": (example / "reference.py").read_text(),
        "utils.py": (repo / "examples" / "utils.py").read_text(),
        "eval.py": (repo / "examples" / "eval.py").read_text(),
    }
    tests = (
        "\n".join(f"size: {size}; seed: 4242" for size in (1, 127, 128, 129, 256, 512, 1024)) + "\n"
    )
    started = time.perf_counter()
    built = compile_cpu.remote(sources, ARCH)
    artifacts = built.pop("artifacts")
    print("CPU_BUILD", json.dumps(built), flush=True)
    print("ARTIFACT_BYTES", sum(map(len, artifacts.values())), flush=True)
    evaluated = evaluate_gpu.remote(sources, artifacts, ARCH, tests)
    report = {
        "requested_gpu": GPU,
        "target_arch": ARCH,
        "artifact_bytes": sum(map(len, artifacts.values())),
        "cpu_build": built,
        "gpu": evaluated,
        "client_pipeline_seconds": time.perf_counter() - started,
    }
    print("RESULT", json.dumps(report, indent=2), flush=True)
    if output:
        Path(output).write_text(json.dumps(report, indent=2) + "\n")
