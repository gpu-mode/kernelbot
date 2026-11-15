# Modal Torch header cache

The runner sets `CXX` to a GCC wrapper that loads a precompiled
`torch/extension.h` from a Modal Volume. Ordinary `load_inline()` calls need no
changes. Leave `use_pch=False`; PyTorch's separate native PCH option is unnecessary.
The submission path assumes `load_inline()`.

NVCC also caches the implicit CUDA-side framework headers by default (see below).

The host cache only precompiles the Torch host header. CUDA/CUTLASS templates, user headers, kernel
objects, and shared libraries still compile in each submission container.
Different sources and module names share a PCH when their host flags match;
changing a user header after the Torch include still changes the compiled code.

## Profiles and fallback

The CPU warmer builds CPU-only and CUDA-linked profiles for these `extra_cflags`:
`[]`, `["-O2"]`, `["-O3"]`, and `["-O3", "-ffast-math"]`.
Keys include installed headers, compiler, Python/Torch versions, host flags, and
include environment. CUDA-only flags do not change the host key.

Unknown profiles compile normally; sources without the initial Torch include
bypass PCH. With `no_implicit_headers=True`, explicitly include
`<torch/extension.h>` first in C++ to use the cache. Python/Triton code that does
not invoke `CXX` is unaffected.

Misses and bypasses still pay wrapper startup cost. There is no guarantee of a
speedup. Set `CXX=/usr/bin/g++` before building to bypass the wrapper entirely.
`KERNELBOT_PCH_DISABLE=1` disables cache lookup but retains wrapper startup.

## Deployment

### CUDA-side framework cache

The runner uses NVIDIA's checksum-pinned NVCC 13.4.99 drop with the complete CUDA
13.4 toolkit. `PYTORCH_NVCC` automatically reuses `torch/types.h`, `cuda.h`, and
`cuda_runtime.h` from `kernelbot-nvcc-pch`. CUTLASS and user headers stay outside
this cache. The trusted warmer builds 17 profiles from five observed flag tuples;
submission workers mount them read-only. Unknown flags, custom compiler inputs,
and sources without the implicit header prefix compile normally. Corrupt caches
fall back to normal compilation. CPU artifact replay checks compiler identity
and user-header dependencies.

Upload the private compiler drop once, without committing binaries:

```sh
PYTHONPATH=src/runners uv run python scripts/upload_nvcc_toolchain.py --source /private/path/to/binaries
```

CI warms both caches before deployment. `KERNELBOT_NVCC_PCH_PROFILES` selects a
comma-separated subset; empty warms all 17. Manual warmup uses
`modal run src/runners/modal_runner.py::warm_nvcc_pch --profiles ...`.
Set `KERNELBOT_NVCC_PCH=0` to disable device PCH. Set both
`KERNELBOT_NVCC_TOOLCHAIN=0` and `KERNELBOT_NVCC_PCH=0` to restore CUDA 13.3.
`KERNELBOT_NVCC_PCH_DISABLE=1` also disables lookup within a submission.
The corresponding `KERNELBOT_NVCC_TOOLCHAIN_VOLUME`, `KERNELBOT_NVCC_PCH_VOLUME`,
and `KERNELBOT_PCH_VOLUME` variables override the private input, device, and host
volumes. Retain old cache generations for rollback and replace workers after
warming so they see the new Volume snapshot.

### Host cache

The main/dev workflow warms profiles before deploying GPU runners. Manually,
run both commands in the same Modal profile/environment:

```sh
PYTHONPATH=src:src/runners uv run modal run src/runners/modal_runner.py::warm_pch
PYTHONPATH=src:src/runners uv run modal deploy src/runners/modal_runner_archs.py
```

`KERNELBOT_PCH_VOLUME` overrides the default `kernelbot-torch-pch` Volume.
`warm_pch --profiles cuda-default,cuda-O3` warms selected profiles.
Rewarm after changing the image or installed headers, then replace existing
runner containers to refresh their Volume snapshots.

The CPU warmer accepts fixed profiles and commits completed headers. GPU runners
mount the Volume read-only and use `restrict_modal_access=True`; otherwise the
Modal upload API could bypass the read-only mount. Restricted runners cannot
reload the Volume themselves. GCC validates a PCH before using it.

## Benchmark

```sh
PYTHONPATH=src:src/runners KERNELBOT_PCH_VOLUME=kernelbot-pch-test \
  uv run modal run scripts/modal_pch_benchmark.py
# Actual tensor-core GEMM, two tile variants, two repetitions per mode:
PYTHONPATH=src:src/runners KERNELBOT_PCH_VOLUME=kernelbot-pch-test \
  uv run modal run scripts/modal_pch_benchmark.py --cases cutlass_gemm,cutlass_gemm_implicit --repeats 2
```

Every build runs in a fresh T4 container (4 CPU cores, 16 GiB), with normal Ninja
parallelism. The baseline uses `/usr/bin/g++`; both modes use the same timing
instrumentation. GCC header tracing verifies actual PCH use in a separate,
untimed compilation. The tests check changed headers and source against distinct
expected outputs. The script also retains file-based and plain C++ diagnostics;
these are not representative submission paths.

The CUTLASS case instantiates SM75 tensor-core GEMMs with FP16 inputs, FP32
accumulation/output, 128×128×32 and 128×64×32 threadblock tiles. It checks
`alpha * A @ B.T` for random and zero inputs, tile tails, and a non-default CUDA
stream against a float64 PyTorch reference (`rtol=atol=2e-4`). This measures compile
time, not GEMM execution speed.

## Results

Measured on CUDA 13.3 / PyTorch 2.12.0+cu130, T4. Values are medians from separate
baseline and cached containers; two variants per case, with two repetitions for
CUTLASS. The table covers `load_inline()` builds.

| Build | Baseline | Cached |
| --- | ---: | ---: |
| Minimal CUDA headers | 18.85s | 10.00s |
| CUTLASS GEMM, default implicit CUDA headers | 48.08s | 45.61s |
| CUTLASS GEMM, `no_implicit_headers=True` | 19.95s | 12.58s |
| Unwarmed host flags | 18.08s | 20.30s |

With default implicit CUDA headers, the CUTLASS GEMM improved only **1.05×**,
with overlapping baseline/cached timing ranges. With `no_implicit_headers=True`,
it improved **1.59×** (37% less time). In that case, host compilation fell from
19.10s to 7.88s, while NVCC took 12.54s and 11.53s and became the bottleneck.
The PCH does not cache CUDA headers or templates. Keeping Torch headers out of
the CUDA translation unit produces the larger gain for this GEMM; the fixture
keeps Tensor handling in C++ and explicitly includes its required headers.

All 160 GEMM checks passed, with maximum absolute error 8.53e-6. The remaining
cases, including the supplemental diagnostics, passed another 76 checks.
All eight warmup profiles also completed and
committed on this image.

The `load_inline()` case with unwarmed host flags was slower in this sample.
Separate containers introduce timing variability, so the measured difference
does not isolate wrapper overhead. Larger CUDA builds may hide the host-side
savings entirely.
[Per-container results and Modal runs](benchmarks/modal-pch.json).
