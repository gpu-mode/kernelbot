"""Validate saved measurements and produce a compact, shareable evidence file."""

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def compact_job(job, workload):
    """Retain timings, correctness, and extension provenance without compiler logs."""
    result = job["result"]["runs"]["benchmark"]["run"]["result"]
    assert int(result["benchmark-count"]) == 3
    if job["passed"]:
        assert job["returncode"] == 0 and result["check"] == "pass"
        # Successful shapes log statistics; only failing shapes have a status key.
        assert all(f"benchmark.{i}.status" not in result for i in range(3))
        assert all(
            math.isfinite(float(result[f"benchmark.{i}.mean"])) and float(result[f"benchmark.{i}.mean"]) > 0
            for i in range(3)
        )
        assert len(job["phases"]) == 2
        assert all(phase["exit_code"] == 0 for phase in job["phases"])
    if workload == "inline":
        assert job["compiled_extensions"], "Cold inline job must produce a native library"
    return {
        **{key: job[key] for key in ("index", "workload", "mode", "wall_s", "passed", "compiled_extensions")},
        "phases": [{k: p[k] for k in ("phase", "wall_s", "exit_code")} for p in job["phases"]],
        "benchmark": result,
    }


def summarize(raw):
    """Audit correctness and keep every observation behind the aggregate medians."""
    assert raw["passed"] and raw["reload"]["passed"]
    assert raw["reload"]["observed"] == [True, False, True]
    assert len(raw["batches"]) == raw["arguments"]["rounds"] * 2
    workload = raw["arguments"]["workload"]
    batches = []
    for batch in raw["batches"]:
        assert batch["passed"] and len(batch["jobs"]) == raw["arguments"]["count"]
        batches.append(
            {
                **{k: batch[k] for k in ("index", "mode", "wall_s", "passed")},
                "jobs": [compact_job(j, workload) for j in batch["jobs"]],
            }
        )
    medians = {}
    for mode in ("fresh", "persistent"):
        selected = [b for b in batches if b["mode"] == mode]
        jobs = [j for b in selected for j in b["jobs"]]
        medians[mode] = {
            "batch_wall_s": statistics.median(b["wall_s"] for b in selected),
            "job_wall_s": statistics.median(j["wall_s"] for j in jobs),
            "phase_wall_s": {
                phase: statistics.median(p["wall_s"] for j in jobs for p in j["phases"] if p["phase"] == phase)
                for phase in ("compile-import", "benchmark")
            },
            "kernel_mean_us": {
                jobs[0]["benchmark"][f"benchmark.{i}.spec"]: statistics.median(
                    float(j["benchmark"][f"benchmark.{i}.mean"]) / 1000 for j in jobs
                )
                for i in range(3)
            },
        }
    reload_jobs = [compact_job(j, workload) for j in raw["reload"]["jobs"]]
    wrong = reload_jobs[1]["benchmark"]
    assert all(
        wrong[f"benchmark.{i}.status"] == "fail" and "mismatch" in wrong[f"benchmark.{i}.error"] for i in range(3)
    )
    return {
        "passed": True,
        "system": {k: v for k, v in raw["system"].items() if k != "hostname"},
        "arguments": raw["arguments"],
        "medians": medians,
        "wall_time_reduction_pct": 100 * (1 - medians["persistent"]["batch_wall_s"] / medians["fresh"]["batch_wall_s"]),
        "batches": batches,
        "reload": {"observed": raw["reload"]["observed"], "jobs": reload_jobs},
    }


def main():
    """Read the complete local result and write audited JSON to stdout."""
    parser = argparse.ArgumentParser()
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    data = args.results.read_bytes()
    summary = summarize(json.loads(data))
    summary["raw_results_sha256"] = hashlib.sha256(data).hexdigest()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
