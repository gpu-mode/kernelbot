# NVCC framework header cache

KernelBot can use NVIDIA's private PCH compiler drop to precompile the framework
headers that PyTorch inserts into `load_inline()` CUDA sources. Matching builds
reuse those headers from a shared, read-only Modal Volume. Users keep their
existing submissions and compiler flags.

This is a separately selectable CUDA 13.4 toolkit on the existing runner base
image. Both the toolchain and the NVCC cache are disabled by default. The existing
[host PCH cache](modal-pch-cache.md)
and [CPU-to-GPU artifact transfer](python-cpu-compilation.md) work alongside it.
The [initial benchmark](nvcc-pch-benchmark.md) used broader, workload-specific
headers, including a CUTLASS header. Its speedups do not measure this narrower
framework-only cache. [Release validation results](benchmarks/nvcc-pch-release.json)
record measurements for the implemented CPU-to-GPU path.

## Compatibility

The forwarding header contains, in order, `torch/types.h`, `cuda.h`, and
`cuda_runtime.h`. These are the exact implicit CUDA includes in the pinned
PyTorch 2.12 image; `torch/types.h` includes ATen transitively. CUTLASS and user
headers are not precompiled. Adding arbitrary common headers before user code
could change macro and include-order behavior, so header-frequency data alone
does not make an include eligible.

The wrapper requires this source prefix and an exact match for the compiler
flags and header-search environment. It preserves flag order, macros, and the
real module name in the compiler invocation. Cache lookup omits generated
source/output/dependency paths and the per-module name macro.
Unknown profiles, custom include paths, compiler overrides, and sources without
the prefix compile normally. `no_implicit_headers=True` does not gain injected
framework headers. A rejected cache retries the original compile once; genuine
submission errors still fail.

The compiler is NVIDIA's `13.4.99` drop, paired with toolkit package `13.4.3-1`
and CUDA 13.4.92 compiler packages. The complete toolkit preserves cuTile,
cuBLAS, sanitizer, and other existing development tools when CUDA paths switch.
CUDA 13.3's `ptxas` cannot assemble the emitted PTX 9.4. The installer
verifies all four supplied binary hashes and explicitly selects `nvvm-next` for
the supported GPUs. Image-build smoke checks cover every deployed architecture
and the architecture-specific Blackwell profiles.

## Observed profiles and candidate coverage

The [profile inventory](benchmarks/nvcc-pch-profiles.json) covers the
2026-11-13 private-diff snapshot: **366,803 unique submissions** in 367,118 rows.
The public baseline is excluded; this is not the complete GPUMODE submission
history. Runner metadata identifies 124,736 NVIDIA submissions, leaves 242,065
missing or unrecognized, and identifies two other runners. Those cohorts remain
separate throughout the analysis.

A conservative static pass examines calls with an explicit `cuda_sources`
keyword. It resolves simple literals and unambiguous assignments in their lexical
scope, rejecting reassigned, mutated, escaped, or otherwise unresolved values.
Calls using `**kwargs` are never assumed to have default flags. An eligible
candidate also needs a verified PyTorch `load_inline` import, resolved CUDA
source, implicit headers enabled, no custom include paths, and flags from the
bounded inventory allowlist. Flags retain their original order. This finds
20,411 eligible candidate submissions among 70,461 static CUDA `load_inline`
submissions with known NVIDIA runners, and 25,996 among 112,520 with unknown
runners.

The default catalog uses the six most frequent eligible ordered tuples in the
**known NVIDIA cohort**:

| Ordered `extra_cuda_cflags` | Known NVIDIA submissions | Unknown-runner submissions |
| --- | ---: | ---: |
| `['-O3']` | 5,419 | 10,457 |
| `['-O3', '--use_fast_math']` | 3,358 | 6,695 |
| `[]` | 2,670 | 2,556 |
| `['-O3', '--use_fast_math', '-arch=sm_100a']` | 2,609 | 113 |
| `['-O3', '-arch=sm_100', '--use_fast_math']` | 771 | 99 |
| `['-O3', '-std=c++17', '-arch=sm_80', '--use_fast_math']` | 628 | 174 |

The three architecture-neutral tuples are warmed for `sm75`, `sm80`, `sm89`,
`sm90a`, and `sm100`; the three explicit-architecture tuples add one profile each,
for **18 default profiles**. The warmer captures PyTorch's remaining generated
flags. Different ordering, include paths, or generated architecture flags still
require their own exact profile.

Each table row counts a submission once per distinct tuple. The coverage below
uses a set union per submission, so a submission containing several selected
tuples is counted once:

| Runner cohort | Static CUDA `load_inline` submissions | Any eligible call with a selected tuple | All observed calls eligible and selected |
| --- | ---: | ---: | ---: |
| Known NVIDIA | 70,461 | 15,421 (21.89%) | 15,176 |
| Unknown | 112,520 | 18,283 (16.25%) | 16,356 |
| Combined analyzed cohorts | 182,981 | 33,704 (18.42%) | 31,532 |

These are **static flag-tuple candidate counts**, not executed NVCC command
counts or measured cache-hit rates. Matching the real architecture, framework
prefix, toolchain, and environment remains necessary. Dynamic Python and
positional CUDA arguments fall outside this inventory's coverage. In particular,
88,425 analyzed submissions contain a call with `no_implicit_headers=True`; those
calls keep their existing behavior and do not receive the master header.

Resolved source includes also show why header popularity cannot define the
master header. The following frequencies count known-NVIDIA static CUDA
`load_inline` submissions once per explicit include across their resolved C++
and CUDA source strings; they do not count PyTorch's generated implicit includes:

| Explicit source include | Submissions | In the master header |
| --- | ---: | --- |
| `cuda_runtime.h` | 50,015 | Yes |
| `cuda_fp16.h` | 38,863 | No |
| `cuda.h` | 35,932 | Yes |
| `torch/extension.h` | 24,752 | No |
| `ATen/ATen.h` | 3,448 | Transitive through `torch/types.h` |
| `cutlass/cutlass.h` | 263 | No; CUTLASS excluded |
| `cute/tensor.hpp` | 232 | No; CuTe excluded |
| `torch/types.h` | 159 | Yes; normally inserted by PyTorch |

Only the exact three-header implicit prefix described above is precompiled.
The published inventory contains aggregates, without submission IDs or private
source code.

## Immutable cache and artifacts

The cache layout is
`/kernelbot-nvcc-pch/<toolchain fingerprint>/<profile key>/`.
Each completed entry contains `gpumode.cuh`, both frontend PCH files, and a ready
manifest. The toolchain fingerprint covers the compiler drop, supporting CUDA
tools, installed CUDA package versions and headers, and the wrapper and warmer
sources. Profile keys add the ordered flags and relevant environment.

Only the trusted CPU warmer can write the Volume. Submission workers mount it
read-only with Modal API access restricted. The warmer accepts fixed profile
names rather than submission-supplied code or flags, and publishes a manifest
only after both frontend files are complete. New generations coexist with old
ones so an image rollback retains its matching cache.

CPU artifact packets include the NVCC toolchain identity. GPU replay checks that
identity and ordinary source/header dependencies. A PCH dependency uses its small
trusted manifest and file sizes instead of reading several GB to hash it on each
submission. Arbitrary PCH paths are not eligible for this shortcut. Changed user
headers and compiler environment variables still invalidate replay.

## Prepare and validate a release

Use KernelBot's Modal workspace. Keep the NVIDIA binaries outside this public
repository. The input directory must contain `nvcc`, `cudafe++`,
`nvvm/bin/cicc`, and `nvvm-next/bin/cicc`.

```sh
export PYTHONPATH=src:src/runners
uv run python scripts/upload_nvcc_toolchain.py --source /private/path/to/binaries
uv run modal run scripts/validate_nvcc_pch.py --output /tmp/nvcc-pch-validation.json
```

The validator selects the new toolchain and isolated cache Volumes. Its default
matrix runs PCH on/off on T4, H100, and B200, plus T4 checks for unknown flags,
missing/corrupt caches, disabled implicit headers, changed user headers, and a
custom NVCC override. It exercises real CPU compilation, GPU artifact replay,
correctness, benchmarking, and ranking evaluation without publishing scores.
Use `--profiles` to select trusted catalog entries and `--repeats` for repeated
PCH-on/off pairs in fresh containers. Successful compilation alone is not a
cache hit: the validator requires both NVIDIA frontends to report consumption.
The [profile inventory](benchmarks/nvcc-pch-profiles.json) describes the sampled
submission population and static candidate coverage. Those counts are not
executed NVCC command counts or measured cache-hit rates.

## Deploy and roll back

First warm and deploy a separate canary app using a single consistent image:

```sh
export KERNELBOT_MODAL_APP=discord-bot-runner-nvcc-pch-canary
export KERNELBOT_NVCC_TOOLCHAIN=1
export KERNELBOT_NVCC_PCH=1
uv run modal run src/runners/modal_runner.py::warm_pch
uv run modal run src/runners/modal_runner.py::warm_nvcc_pch
uv run modal deploy src/runners/modal_runner_archs.py
```

`ModalLauncher(app_name=...)` can target that app for non-scoring validation.
Do not split one scoring population randomly across compiler versions. Promote
the validated toolchain as a consistent runner release, recording the image and
toolchain fingerprint with the rollout.

The main/dev deployment workflow reads these repository/environment variables:

| Variable | Default | Purpose |
| --- | --- | --- |
| `KERNELBOT_NVCC_TOOLCHAIN` | `0` | Select the pinned CUDA 13.4 compiler image. |
| `KERNELBOT_NVCC_PCH` | `0` | Enable automatic NVCC PCH lookup. Requires the toolchain. |
| `KERNELBOT_NVCC_PCH_PROFILES` | empty | Comma-separated trusted profiles; empty uses the bounded default set. |
| `KERNELBOT_NVCC_TOOLCHAIN_VOLUME` | `kernelbot-nvcc-toolchains` | Private compiler input Volume. |
| `KERNELBOT_NVCC_PCH_VOLUME` | `kernelbot-nvcc-pch` | Device PCH Volume. |
| `KERNELBOT_PCH_VOLUME` | `kernelbot-torch-pch` | Host PCH Volume. |

`warm_nvcc_pch --profiles ...` limits manual warmup to specific catalog entries.
Warm before replacing workers, since existing containers retain their Volume
snapshot.

For a cache-only rollback, deploy the same toolchain with
`KERNELBOT_NVCC_PCH=0`. For a compiler rollback, deploy with both switches `0` to
restore the CUDA 13.3 image. Individual submissions can disable lookup with
`KERNELBOT_NVCC_PCH_DISABLE=1` or override `PYTORCH_NVCC`.

Compilation emits `[kernelbot-nvcc-pch]` JSON events for `hit`, `miss`, `bypass`,
`disabled`, `fallback`, or `error`. A hit requires NVIDIA diagnostics confirming
both `cicc` and `cudafe` consumed the cache. Events include profile identity and
elapsed compiler time. `KERNELBOT_NVCC_PCH_EVENTS` optionally writes JSONL to a
container-local file.
