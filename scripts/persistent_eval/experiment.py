"""Compare fresh and persistent processes with cold per-submission caches."""

import argparse
import contextlib
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

from export_result import emit_result
from persistent import running_server


def run_job(root, index, workload, mode):
    """Use a new coordinator and private build directory for each submission."""
    directory = root / f"job-{index}-{workload}"
    directory.mkdir(parents=True)
    env = dict(
        os.environ,
        MAX_JOBS="2",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        TORCH_CUDA_ARCH_LIST="7.5",
        TORCH_EXTENSIONS_DIR=str(directory / "torch-cache"),
        TRITON_CACHE_DIR=str(directory / "triton-cache"),
        CUDA_CACHE_PATH=str(directory / "cuda-cache"),
    )
    started = time.monotonic()
    child = subprocess.run(
        [sys.executable, "/experiment/worker.py", f"/experiment/{workload}.json", mode],
        cwd=directory,
        env=env,
        capture_output=True,
        text=True,
        timeout=420,
    )
    wall_s = time.monotonic() - started
    result_file = directory / "result.json"
    row = json.loads(result_file.read_text()) if result_file.exists() else {"passed": False}
    artifacts = [
        {
            "name": str(p.relative_to(directory)),
            "bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
        }
        for p in sorted(directory.rglob("*.so"))
    ]
    row.update(
        index=index,
        workload=workload,
        mode=mode,
        wall_s=wall_s,
        returncode=child.returncode,
        stdout=child.stdout,
        stderr=child.stderr,
        compiled_extensions=artifacts,
    )
    print("JOB", json.dumps({k: row[k] for k in ("index", "workload", "mode", "wall_s", "passed")}), flush=True)
    return row


def run_batch(root, index, mode, workload, count):
    """Include worker startup and teardown; GPU measurements stay sequential."""
    started = time.monotonic()
    directory = root / f"batch-{index}-{mode}"
    server = running_server() if mode == "persistent" else contextlib.nullcontext()
    with server:
        jobs = [run_job(directory, i, workload, mode) for i in range(count)]
    result = {
        "index": index,
        "mode": mode,
        "wall_s": time.monotonic() - started,
        "passed": all(j["passed"] for j in jobs),
        "jobs": jobs,
    }
    print("BATCH", json.dumps({k: v for k, v in result.items() if k != "jobs"}), flush=True)
    return result


def check_reload(root, workload):
    """Require correct/wrong/correct with the same inline extension name."""
    wrong = json.loads(Path(f"/experiment/{workload}.json").read_text())
    before, after = (
        ("C[idx] = A[idx] + B[idx]", "C[idx] = A[idx] - B[idx]") if workload == "inline" else ("C = A + B", "C = A - B")
    )
    assert wrong["sources"]["submission.py"].count(before) == 1
    wrong["sources"]["submission.py"] = wrong["sources"]["submission.py"].replace(before, after)
    Path("/experiment/incorrect.json").write_text(json.dumps(wrong))
    with running_server():
        jobs = [
            run_job(root / "reload", i, name, "persistent") for i, name in enumerate([workload, "incorrect", workload])
        ]
    observed = [j["passed"] for j in jobs]
    # A compiler/import failure is not evidence of detecting incorrect math.
    wrong_runs = jobs[1].get("result", {}).get("runs", {})
    wrong_results = [r.get("run", {}) for r in wrong_runs.values()]
    correctness_rejected = any(r and r.get("result", {}).get("check") == "fail" for r in wrong_results)
    result = {
        "expected": [True, False, True],
        "observed": observed,
        "jobs": jobs,
        "correctness_rejected": correctness_rejected,
        "passed": observed == [True, False, True] and correctness_rejected,
    }
    print("RELOAD", json.dumps({k: v for k, v in result.items() if k != "jobs"}), flush=True)
    return result


def main():
    """Run counterbalanced rounds and export checksummed complete evidence."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", choices=["inline", "triton"], default="inline")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args()
    assert args.count > 0 and args.rounds > 0
    root = Path("/experiment/output")
    root.mkdir()
    info = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import json,dataclasses; from libkernelbot.run_eval import make_system_info; "
            "print(json.dumps(dataclasses.asdict(make_system_info())))",
        ],
        text=True,
    )
    Path("/experiment/system.json").write_text(info)
    assert json.loads(info)["device_count"] == 1
    hardware = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total", "--format=csv"], text=True
    )
    print("HARDWARE", hardware.strip(), flush=True)
    batches = []
    for round_index in range(args.rounds):
        order = ["fresh", "persistent"] if round_index % 2 == 0 else ["persistent", "fresh"]
        for mode in order:
            batches.append(run_batch(root, len(batches), mode, args.workload, args.count))
            emit_result({"checkpoint": "batch", "batch": batches[-1]})
            if not batches[-1]["passed"]:
                break
        if not batches[-1]["passed"]:
            break
    reload_check = check_reload(root, args.workload) if all(b["passed"] for b in batches) else None
    summary = {
        mode: statistics.median(b["wall_s"] for b in batches if b["mode"] == mode)
        for mode in {b["mode"] for b in batches}
    }
    result = {
        "system": json.loads(info),
        "hardware": hardware,
        "arguments": vars(args),
        "batches": batches,
        "summary": summary,
        "reload": reload_check,
        "passed": all(b["passed"] for b in batches) and bool(reload_check and reload_check["passed"]),
    }
    print("SUMMARY", json.dumps(summary), flush=True)
    emit_result(result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
