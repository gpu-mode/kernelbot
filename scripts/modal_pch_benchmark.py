"""Real submissions in fresh Modal containers; no production deployment required.

PYTHONPATH=src:src/runners KERNELBOT_PCH_VOLUME=kernelbot-pch-test \\
    uv run modal run scripts/modal_pch_benchmark.py
"""

import io
import json
import os
import re
import statistics
import time
from pathlib import Path

import modal
from modal_runner import PCH_MOUNT, cuda_image, modal_run_config, pch_volume, warm_pch

app = modal.App("kernelbot-pch-benchmark")
app.include(warm_pch.app)


@app.function(
    image=cuda_image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=900,
    single_use_containers=True,
    restrict_modal_access=True,
    volumes={PCH_MOUNT: pch_volume.with_mount_options(read_only=True)},
)
def submit(config: dict, enabled: bool):
    os.environ["KERNELBOT_PCH_DISABLE"] = "0" if enabled else "1"
    os.environ["KERNELBOT_PCH_TRACE"] = "1"
    os.environ.pop("MAX_JOBS", None)  # Match normal load_inline/Ninja parallelism.
    started = time.perf_counter()
    result = modal_run_config(config)
    assert result.success, result.error
    test = result.runs["test"]
    assert test.run.success and test.run.passed, test.run
    assert test.compilation and test.compilation.exit_code == 0, test.compilation
    logs = test.compilation.stdout + test.compilation.stderr
    durations = re.findall(r"PCH_LOAD_INLINE_SECONDS=([0-9.]+)", logs)
    assert durations, logs[-5000:]
    consumed = "! /kernelbot-pch/" in logs
    assert consumed == enabled, logs[-5000:]
    # A read-only cache must remain read-only from submission containers.
    try:
        Path(PCH_MOUNT, "submission-write-probe").write_text("must fail")
    except OSError:
        pass
    else:
        raise AssertionError("Submission can write the shared PCH Volume")
    try:
        with pch_volume.batch_upload() as batch:
            batch.put_file(io.BytesIO(b"must fail"), "/submission-api-write-probe")
    except modal.exception.AuthError:
        pass
    else:
        raise AssertionError("Submission can write the shared PCH Volume through the API")
    return {
        "api_write_denied": True,
        "enabled": enabled,
        "gpu": result.system.gpu,
        "torch": result.system.torch,
        "container_id": os.environ["MODAL_TASK_ID"],
        "image_id": os.environ["MODAL_IMAGE_ID"],
        "load_inline_seconds": float(durations[0]),
        "cpp_seconds": [float(v) for v in re.findall(r"compile_seconds=([0-9.]+)", logs)],
        "pch_consumed": consumed,
        "keys": re.findall(r"\[kernelbot-pch\] hit (\w+)", logs),
        "tests": test.run.result,
        "remote_seconds": time.perf_counter() - started,
    }


@app.local_entrypoint()
def main(output: str = "/tmp/kernelbot-pch-results.json", repeats: int = 2, headers: str = "minimal"):
    if headers not in {"minimal", "implicit"}:
        raise ValueError("headers must be minimal or implicit")
    from libkernelbot.consts import SubmissionMode
    from libkernelbot.task import build_task_config, make_task_definition

    print(warm_pch.remote(profiles="cuda-default"))
    root = Path(__file__).resolve().parents[1] / "examples/vectoradd_py"
    task = make_task_definition(root).task
    filename = "submission_cuda_pch.py" if headers == "minimal" else "submission_cuda_inline.py"
    source = (root / filename).read_text()
    rows = []
    for index in range(repeats):
        for enabled in (False, True):
            variant = source.replace(
                "const int threads = 256", f"const int threads = {128 if index else 256}"
            )
            variant = variant.replace(
                "name='add_cuda'", f"name='pch_variant_{index}_{int(enabled)}'"
            )
            variant = variant.replace(
                "add_module = load_inline(",
                "import time\n_pch_started = time.perf_counter()\nadd_module = load_inline(",
            )
            variant = variant.replace(
                "\ndef add(A, B):",
                '\nprint(f"PCH_LOAD_INLINE_SECONDS={time.perf_counter() - _pch_started:.3f}", flush=True)\n'
                "\ndef add(A, B):",
            )
            config = build_task_config(
                task=task, submission_content=variant, arch="75", mode=SubmissionMode.TEST
            )
            row = submit.remote(config, enabled)
            row["headers"] = headers
            row["variant"] = index
            rows.append(row)
            Path(output).write_text(json.dumps(rows, indent=2))
            print(json.dumps(row), flush=True)
    assert len({row["container_id"] for row in rows}) == len(rows), "Expected fresh containers"
    warm_keys = {key for row in rows if row["enabled"] for key in row["keys"]}
    assert len(warm_keys) == 1, "Different kernel variants must reuse the same header entry"
    cold = statistics.median(row["load_inline_seconds"] for row in rows if not row["enabled"])
    warm = statistics.median(row["load_inline_seconds"] for row in rows if row["enabled"])
    print(f"load_inline median: cold={cold:.3f}s warm={warm:.3f}s speedup={cold / warm:.2f}x")
    cpp_cold = statistics.median(row["cpp_seconds"][0] for row in rows if not row["enabled"])
    cpp_warm = statistics.median(row["cpp_seconds"][0] for row in rows if row["enabled"])
    print(f"C++ median: cold={cpp_cold:.3f}s warm={cpp_warm:.3f}s speedup={cpp_cold / cpp_warm:.2f}x")
    assert cpp_warm < cpp_cold, "PCH did not improve host compilation"
    if headers == "minimal":
        assert warm < cold, "PCH did not improve end-to-end compilation"
