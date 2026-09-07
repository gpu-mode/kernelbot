# Modal Torch header cache

The runner sets `CXX` to a GCC wrapper that loads a precompiled
`torch/extension.h` from a Modal Volume. Ordinary `load_inline()` calls need no
changes. Leave `use_pch=False`; PyTorch's separate native PCH option is unnecessary.
File-based `load()` also benefits when its C++ source starts with the same include.

Only the Torch host header is cached. CUDA/CUTLASS templates, user headers, kernel
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
  uv run modal run scripts/modal_pch_benchmark.py --cases cutlass_gemm --repeats 2
```

Every build runs in a fresh T4 container (4 CPU cores, 16 GiB), with normal Ninja
parallelism. The baseline uses `/usr/bin/g++`; both modes use the same timing
instrumentation. GCC header tracing verifies actual PCH use in a separate,
untimed compilation. The tests check changed headers and source, including
file-based `load()` with the same extension name, against distinct expected outputs.

The CUTLASS case instantiates SM75 tensor-core GEMMs with FP16 inputs, FP32
accumulation/output, 128×128×32 and 128×64×32 threadblock tiles. It checks
`alpha * A @ B.T` for random and zero inputs, tile tails, and a non-default CUDA
stream against a float64 PyTorch reference (`rtol=atol=2e-4`). This measures compile
time, not GEMM execution speed.

## Results

Measured on CUDA 13.3 / PyTorch 2.12.0+cu130, T4. Values are medians from separate
baseline and cached containers; two variants per case, with two repetitions for
CUTLASS. The plain C++ row measures object compilation only.

| Build | Baseline | Cached |
| --- | ---: | ---: |
| Minimal CUDA headers | 18.85s | 10.00s |
| CuTe/MathDx includes (vector add) | 19.94s | 14.24s |
| CUTLASS tensor-core GEMM | 19.95s | 12.58s |
| File-based `load()` | 18.30s | 8.77s |
| Plain C++ | 0.202s | 0.287s |
| Unwarmed host flags | 18.08s | 20.30s |

The CUTLASS GEMM build was **1.59× faster** (37% less time). Host compilation went
from 19.10s to 7.88s; NVCC took 12.54s and 11.53s respectively and became the
bottleneck. The cache does not accelerate NVCC; those separate-container times
include machine variability. All 80 GEMM checks passed, with maximum absolute
error 8.53e-6. The broader cases passed another 100 checks.

The plain C++ wrapper added about 85ms; the miss case was also slower in this
sample. Larger CUDA template builds may hide the host-side savings entirely.
[Per-container results and Modal runs](benchmarks/modal-pch.json).
