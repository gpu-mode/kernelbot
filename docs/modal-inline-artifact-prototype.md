# Compile load_inline submissions before allocating a GPU

This experiment imports the vector-add submission on a CPU-only Modal worker,
copies its compiled `.so` to a T4 worker, and runs KernelBot's correctness harness.
It leaves the submission's `load_inline` call intact and uses the same runner
image for both workers. The production submission path is unchanged.

```bash
uv sync --extra dev
PYTHONPATH=src:src/runners uv run modal run \
  scripts/modal_inline_artifact_prototype.py \
  --output /tmp/inline-artifact-result.json
```

Set `KERNELBOT_PROTOTYPE_GPU` to `L4`, `A100`, `H100`, or `B200` to change the GPU
and compilation target. T4 is the tested default. The app exits after the run.
The compiler wrapper from the runner image is present, but its PCH Volume is
not mounted: both comparison builds start without a precompiled-header cache.

The default input is `examples/vectoradd_py/submission_cuda_inline.py`.
`--submission path/to/submission.py` accepts another implementation of that same
vector-add task. The harness tests square matrices with dimensions 1, 127, 128,
129, 256, 512 and 1024.

## How it works

A `sitecustomize` hook intercepts `load_inline` in the submission process and
the evaluator's spawned processes. On the CPU worker, it calls the compiler and
saves the resulting library and a manifest. On the GPU worker, it checks the
manifest and binary hash, then imports the saved library. Requests are matched
by source, options, target architecture and Python/PyTorch/CUDA ABI information.
Missing artifacts fail; the hook disables PyTorch's Ninja build entrypoint.

The script transfers library bytes through Modal results and arguments. It
checks that replay creates no build directory, runs the existing
`run_single_evaluation` harness, and verifies that a missing artifact fails.
It then compiles once on the GPU worker to provide a timing baseline. That last
build is diagnostic overhead, not part of the proposed CPU-build path.

## Results

The [T4 run](https://modal.com/apps/coreauto/main/ap-kfECmtFhii1jy58mgwXMYY) passed
all seven correctness cases and the missing-artifact check. Replay created no
build directory. Both workers requested 4 CPU cores and 8 GiB RAM.

| Step | Extension compile/load | Python process including startup |
| --- | ---: | ---: |
| CPU build, zero visible GPUs | 55.684 s | 58.793 s |
| T4 artifact load | 7.728 ms | 2.722 s |
| Fresh build on the T4 worker | 55.413 s | 58.042 s |

The transferred library and manifest were 1,560,196 bytes. Evaluation took
5.959 s. These are single-run observations, not throughput or billing estimates.

The JSON output separates extension load/compile time, Python process time,
evaluation time and total experiment time. Compare compilation and replay
separately from scheduling, Python startup and the diagnostic baseline.

## Limits

The submission must compile at import time without querying or using a GPU.
Lazy builds, `load()`, non-Python extensions and libraries outside the common
image are unsupported. Errors propagate without a fallback.

Artifacts are used within one run; this is not a persistent compilation cache.
The manifest does not fingerprint external headers or every compiler input.
Sharing artifacts between submissions would require a complete build identity.
Only use binaries produced by the builder for the same request: the hash detects
corruption, and the Python hook is not a security boundary.

Run the adapter tests without CUDA:

```bash
uv run pytest tests/test_inline_artifacts.py
```
