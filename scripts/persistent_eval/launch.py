"""Build a minimal image and run the trusted-example experiment on one Modal T4."""

import argparse
import base64
import gzip
import hashlib
import json
import shutil
import subprocess
import tempfile
import threading
import time
from pathlib import Path

import modal
import yaml


def prepare_payload(payload, repo):
    """Copy the current evaluator and examples without importing GPU dependencies."""
    scripts = Path(__file__).resolve().parent
    for name in ("worker.py", "persistent.py", "experiment.py", "export_result.py"):
        shutil.copyfile(scripts / name, payload / name)
    library = payload / "source/libkernelbot"
    library.mkdir(parents=True)
    (library / "__init__.py").touch()
    for name in ("run_eval.py", "consts.py"):
        shutil.copyfile(repo / "src/libkernelbot" / name, library / name)
    example = repo / "examples/vectoradd_py"
    task = yaml.safe_load((example / "task.yml").read_text())
    for workload, submission in [("inline", "submission_cuda_inline.py"), ("triton", "submission_triton.py")]:
        sources = {
            entry["name"]: (
                example / (submission if entry["source"] == "@SUBMISSION@" else entry["source"])
            ).read_text()
            for entry in task["files"]
        }
        config = {
            "main": task["config"]["main"],
            "sources": sources,
            "lang": "py",
            "arch": 75,
            "benchmarks": task["benchmarks"][:3],
            "tests": task["tests"],
            "mode": "benchmark",
            "test_timeout": 180,
            "benchmark_timeout": 180,
            "ranked_timeout": 180,
            "ranking_by": "mean",
            "seed": None,
            "multi_gpu": False,
        }
        (payload / f"{workload}.json").write_text(json.dumps(config))
    return {
        str(p.relative_to(payload)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(payload.rglob("*"))
        if p.is_file()
    }


def decode_result(chunks, expected_sha256):
    """Reject incomplete or corrupted exported evidence."""
    raw = gzip.decompress(base64.b64decode("".join(chunks)))
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("Result checksum mismatch")
    return json.loads(raw)


def collect(sandbox, output):
    """Save logs and reconstruct a complete result despite Modal's line limit."""
    errors = []

    def stderr_reader():
        try:
            with (output / "stderr.log").open("w") as log:
                for line in sandbox.stderr:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
        except Exception as error:
            errors.append(error)

    reader = threading.Thread(target=stderr_reader, daemon=True)
    reader.start()
    chunks = []
    result = None
    with (output / "stdout.log").open("w") as log:
        for line in sandbox.stdout:
            log.write(line)
            log.flush()
            if line.startswith("EXPERIMENT_CHUNK="):
                chunks.append(line.split("=", 1)[1].strip())
            elif line.startswith("EXPERIMENT_END="):
                result = decode_result(chunks, line.split("=", 1)[1].strip())
                (output / "results.json").write_text(json.dumps(result, indent=2))
                print("SAVED", str(output / "results.json"), flush=True)
            else:
                print(line, end="", flush=True)
    sandbox.wait()
    reader.join(timeout=10)
    if errors:
        raise RuntimeError("Failed to preserve stderr") from errors[0]
    if reader.is_alive():
        raise RuntimeError("stderr stream did not finish")
    if result is None:
        raise RuntimeError("Sandbox exited without complete results; see saved logs")
    return result


def main():
    """Run an ephemeral sandbox and retain local evidence before cleanup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workload", choices=["inline", "triton"], default="inline")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.count < 1 or args.rounds < 1:
        parser.error("count and rounds must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory(prefix="kernelbot-persistent-") as temporary:
        payload = Path(temporary)
        hashes = prepare_payload(payload, repo)
        manifest = {
            "kernelbot_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "payload_sha256": hashes,
            "arguments": {**vars(args), "output_dir": str(output)},
            "image": "nvidia/cuda:12.8.1-devel-ubuntu22.04",
            "python": "3.11",
            "packages": ["torch==2.7.1", "numpy==2.2.6", "ninja==1.11.1.4"],
            "gpu": "T4",
            "cpu": 8,
            "memory_mib": 32768,
        }
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
        image = (
            modal.Image.from_registry(manifest["image"], add_python="3.11")
            .apt_install("g++")
            .pip_install(*manifest["packages"])
            .add_local_dir(payload, "/experiment", copy=True)
            .env({"PYTHONPATH": "/experiment:/experiment/source", "PYTHONUNBUFFERED": "1"})
        )
        command = [
            "python",
            "/experiment/experiment.py",
            "--workload",
            args.workload,
            "--count",
            str(args.count),
            "--rounds",
            str(args.rounds),
        ]
        app = modal.App("kernelbot-persistent-evaluator-experiment")
        with modal.enable_output(), app.run():
            started = time.monotonic()
            sandbox = modal.Sandbox.create(*command, image=image, app=app, gpu="T4", cpu=8, memory=32768, timeout=1800)
            print("SANDBOX", sandbox.object_id, flush=True)
            metadata = {"id": sandbox.object_id, "command": command}
            (output / "sandbox.json").write_text(json.dumps(metadata, indent=2))
            try:
                result = collect(sandbox, output)
                return 0 if sandbox.returncode == 0 and result["passed"] else 1
            finally:
                elapsed = time.monotonic() - started
                sandbox.terminate()
                sandbox.detach()
                metadata.update(elapsed_s=elapsed, stopped=True, gpu_cost_estimate_usd=elapsed * 0.000164)
                (output / "sandbox.json").write_text(json.dumps(metadata, indent=2))
                print("ELAPSED", elapsed, "GPU_COST_ESTIMATE_USD", elapsed * 0.000164, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
