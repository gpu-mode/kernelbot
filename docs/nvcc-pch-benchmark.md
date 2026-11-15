# Experimental NVCC PCH benchmark

`scripts/nvcc_pch_benchmark.py` tests NVIDIA's supplied PCH compiler against
ordinary `torch.utils.cpp_extension.load_inline()` builds. The existing
[Modal PCH cache](modal-pch-cache.md) precompiles the host-side Torch header with
GCC. This experiment adds NVCC header reuse from a separate shared Modal Volume.
It measures compilation latency; it does not optimize kernel execution.

Measured on 2026-11-15: adding the supplied compiler and shared NVCC PCH to
an existing host-PCH setup reduced median build time **31.7% for the simple
implicit-header case (35.37s → 24.17s)** and **22.2% for CUTLASS
(43.72s → 34.03s)** on the same CUDA 13.4 toolkit. This is a useful incremental
improvement, approximately **1.46× and 1.29×**, with substantial CUTLASS timing
variation. Replacing the compiler without enabling its PCH did not improve
these medians. The experiment did not modify production.

These are results for the original workload-specific forwarding headers. The
simple profile also precompiled `cuda_fp16.h`; the CUTLASS profile precompiled
`cutlass/gemm/device/gemm.h`. The [shipping runtime](nvcc-pch-cache.md) caches
only PyTorch's three implicit CUDA headers and leaves CUTLASS headers in each
submission. These POC speedups do not establish the final runtime's speedup.
Use the [release validation results](benchmarks/nvcc-pch-release.json) for that
separate implementation and execution path.

## Toolkit compatibility

The supplied compiler reports CUDA **13.4.99**. The initial attempt to overlay
its four binaries on KernelBot's CUDA 13.3 image failed: the compiler emitted
PTX 9.4, while the installed assembler supported PTX 9.3. Those failures are
compatibility results, not PCH performance measurements.

The benchmark therefore installs the official CUDA 13.4 compiler/toolkit
dependencies, which resolved to compiler **13.4.92**, before applying the
experimental binaries. The stock and patched timing arms use this same 13.4
environment. The official `cuda-nvcc-13-4` package supplies matching `ptxas`,
`nvlink`, and `fatbinary`; `libnvvm-13-4` supplies `cicc` and `libdevice`.
The JSON probe records the actual stock and experimental compiler versions.

Both the official 13.3 and 13.4 `nvcc.profile` files select `nvvm/bin/cicc`.
Copying the supplied second backend to `nvvm-next/bin/cicc` does not select it.
For this bounded SM75/SM100 experiment, patched containers explicitly set:

```text
CICC_PATH = $(TOP)/nvvm-next/bin
```

The probe checks the dispatched path and compiles a minimal kernel for both
`sm_75` and `sm_100`. SM100 is a **compile-only Blackwell smoke test**; it is
not a Blackwell runtime or performance measurement. The separate
[runtime installer](nvcc-pch-cache.md) checks all supported runner architectures;
this experiment makes no claim about older targets or their backend dispatch.

## Comparison

| Mode | Compiler | GCC host PCH | NVCC PCH |
| --- | --- | --- | --- |
| `stock_no_pch` | Official 13.4 | Off | Off |
| `stock_host_pch` | Official 13.4 | On | Off |
| `patched_host_pch` | Experimental 13.4.99 | On | Off |
| `patched_all_pch` | Experimental 13.4.99 | On | On |

Compare `patched_all_pch` with `patched_host_pch` to isolate the added NVCC PCH
path. Compare it with `stock_host_pch` to estimate the combined effect of the
compiler replacement and NVCC PCH on the existing host-cache setup.

Each timed build uses a fresh T4 container with four CPU cores, 16 GiB RAM, a
fresh build directory, and normal Ninja parallelism. Both workloads retain
PyTorch's default implicit CUDA headers:

- `simple`: FP16 vector addition, with two launch/source variants.
- `cutlass`: an actual SM75 tensor-core GEMM, with two threadblock tile variants.

Both variants change a user header and have distinct expected outputs. The
GEMM checks random and zero inputs, tile tails, and a non-default CUDA stream
against a float64 PyTorch reference. Each trial records `load_inline()` wall
time, individual C++/NVCC command times, correctness results, compiler versions,
binary hashes, and container/image identities. Warmup and diagnostic builds are
outside the reported build time. The command order alternates between variants
and repetitions to reduce ordering bias.

Untimed diagnostic compilations verify actual cache consumption: GCC `-H`
must report a valid PCH, and NVCC `--pch-verbose` must report using both
`cudafe_` and `cicc_` PCH files from the expected read-only directory without
creating one. The actual timed-build logs independently confirm both hits. A successful build by itself is insufficient evidence of reuse.

## Run

Use an authenticated Modal environment. Extract the supplied package privately;
the directory below must contain `nvcc`, `cudafe++`, `nvvm/bin/cicc`, and
`nvvm-next/bin/cicc`. Do not commit the supplied binaries or credentials.

```sh
export NVIDIA_PCH_BINARIES=/path/to/extracted/x86_64_Linux_release
export PYTHONPATH=src:src/runners:scripts

# Optional: choose fresh, dedicated benchmark volumes.
export KERNELBOT_HOST_PCH_BENCHMARK_VOLUME=kernelbot-nvcc-host-pch-my-test
export KERNELBOT_NVCC_PCH_VOLUME=kernelbot-nvcc-pch-my-test

# Check compiler versions, backend dispatch, and SM75/SM100 compilation.
uv run modal run scripts/nvcc_pch_benchmark.py \
  --probe-only --output /tmp/nvcc-pch-probe.json

# Warm shared headers, then run all four modes and both source variants.
uv run modal run scripts/nvcc_pch_benchmark.py \
  --cases simple,cutlass --repeats 2 --output /tmp/nvcc-pch-results.json
```

`--warm-only` prepares and commits headers without timing trials. A subsequent
`--skip-warm` run reuses those entries and validates their identity. The harness
uses the temporary `kernelbot-nvcc-pch-benchmark` app and dedicated volumes; it
does not deploy over the production runner. Only the trusted warmer writes the
volumes. Trials mount both caches read-only and restrict Modal API access.

## Integration requirements and limits

In this benchmark, the warmer captures PyTorch's NVCC flags, then compiles a
forwarding `gpumode.cuh` containing the common Torch/CUDA headers and the workload's CUDA
or CUTLASS headers. The header ends with `#pragma hdrstop`. The warmer must specify
`-x cu -c gpumode.cuh -o /tmp/header.o` along with the matched flags and PCH
options; the bare `.cuh` command in the high-level setup note was rejected by
this NVCC build. Trials pass
`-include gpumode.cuh --pch --pch-dir ...`. Cache identity includes supplied
binary/profile hashes, the installed-header fingerprint, forwarding-header
contents, workload flags, and architecture. Exact commands are retained in the
results. The warmup and trials use different extension names, so successful
reuse also tests that per-module naming does not invalidate these profiles.

The separate [runtime](nvcc-pch-cache.md) uses curated submission flag sets,
immutable compiler/header identities, and fallback on a miss. Its release
validator exercises the CPU precompile worker's 8 GiB memory budget and GPU
artifact replay; these original GPU trials used 16 GiB and did not exercise
that path. The runtime also narrows the forwarding header to preserve the
implicit include sequence. User source, kernel objects, and shared libraries
remain outside the shared PCH cache.

This matrix covers two fixtures and one runtime GPU, not the submission
population. It excludes scheduling, image startup, PCH preparation, and the
existing [CPU-to-GPU artifact transfer](python-cpu-compilation.md) from the
`load_inline()` metric. Compiler subprocess times overlap under Ninja and must
not be summed to estimate wall time. The controlled CUDA 13.4 matrix isolates the cache/compiler changes. A
separate CUDA 13.3 baseline below measures the combined toolkit migration and
PCH effect for the same CUTLASS fixture. Fresh containers do not guarantee
cold underlying host or Modal storage caches. Two repetitions
per variant provide a small sample; retain individual measurements alongside
medians and avoid generalizing small differences.

## Results

Main matrix: four builds per cell, two source/tile variants × two repetitions.
Values are median `load_inline()` seconds, with the full min–max range.

| CUDA 13.4 build mode | Simple, implicit headers | CUTLASS, implicit headers |
| --- | ---: | ---: |
| Stock, no PCH | 37.23 (35.75–42.89) | 48.82 (42.95–54.31) |
| Stock + existing host PCH | 35.37 (35.03–36.43) | 43.72 (41.50–48.83) |
| Patched compiler + host PCH | 36.01 (34.74–38.47) | 45.76 (41.73–51.06) |
| Patched compiler + host and NVCC PCH | **24.17 (23.93–29.14)** | **34.03 (28.44–43.66)** |

Comparing the last two rows isolates NVCC PCH on the same patched compiler:
**1.49× simple / 1.34× CUTLASS**. The CUTLASS cached builds took 39–44s in the
first repetition and 28–29s in the second; underlying storage/cache state was
not controlled. The four-sample median is not a guarantee for every fresh
container, and its range overlaps the stock host-cache range.

The fresh production-image CUDA 13.3 baseline measured CUTLASS at **48.75s
without PCH** and **45.32s with the existing host PCH**. Moving from that current
host-cache setup to CUDA 13.4 plus the supplied PCH path measured **34.03s**:
**1.33×, or 24.9% less compile time**, including the toolkit change. The old
benchmark's `inline_simple` case deliberately omits implicit CUDA headers and
is not comparable to the new `simple` fixture; its 10.37s host-cache result is
retained only as a separate minimal-header diagnostic.

A supplemental run tested sequentially reading the PCH files before compilation,
with the read time included in the total. It did not improve the medians:

| Two builds per cell | Direct shared-volume PCH | Prefetch + compilation |
| --- | ---: | ---: |
| Simple | 25.24s | 25.70s |
| CUTLASS | 32.90s | 35.44s |

Prefetch itself took 1.51–1.60s for simple and 2.61–7.73s for CUTLASS. Keep this
supplemental run separate from the main matrix; its faster CUTLASS samples do
not replace the main result.

Warm diagnostic phase timings explain the limit. Combined `cudafe++`/`cicc`
time fell from a median 23.82s to 7.61s for simple and 29.25s to 10.51s for
CUTLASS. GCC preprocessing and compilation still occupied roughly 13–19s.
These are second compilations after timing and correctness checks, so they
illustrate the remaining work rather than measuring first-use storage overhead.

One-time CUDA PCH creation took **44.74s for simple** and **51.80s for CUTLASS**.
Each profile produced two files, totaling **2.52 GB** and **3.00 GB** respectively
(2.35 and 2.79 GiB). The host warmup took 72.26s. These stage timings exclude
volume commits and flag-capture setup. Storage and profile count need budgeting
when expanding to multiple architectures and submission flag sets.

All **492 correctness checks** passed across **62 distinct single-use T4
containers**: 32 main builds, 12 supplemental builds, 16 CUDA 13.3 baseline
builds, and two reuse smoke builds. Maximum GEMM absolute error was **8.53e-6**.
All **18 NVCC-PCH builds** confirmed both frontend hits in their actual timed
logs and in independent diagnostics. Different module names, kernel sources,
and user-header values reused the same profile and produced the expected
distinct outputs. No kernel object or shared-library cache was used.

[Complete measurements, compiler hashes, commands, warmup logs, and validation](benchmarks/nvcc-pch.json).

Modal runs:

- [Main 32-build matrix](https://modal.com/apps/gpu-mode/main/ap-sCbGWCCsGHByigNhMrPUYA)
- [Prefetch and phase diagnostics](https://modal.com/apps/gpu-mode/main/ap-P9owg8O1R3RlDGy0ozOSE1)
- [Production-image CUDA 13.3 baseline](https://modal.com/apps/gpu-mode/main/ap-42e4FtDMIK2OGSSHHcAg51)
- [Shared-volume warmup](https://modal.com/apps/gpu-mode/main/ap-ypnw7T8Xntoe9WGjwycFyh)
- [Cross-module reuse smoke](https://modal.com/apps/gpu-mode/main/ap-ya4TMM2KIT440BrHtTc77E)
