# CPU compilation for Python submissions

Python submissions keep their existing `submission.py` interface. Before allocating
a Modal GPU, the launcher attempts a CPU import for submissions containing
`load_inline`. It transfers any compiled Python extensions to the GPU runner,
which continues through the existing test, benchmark and leaderboard evaluator.
Other Python submissions go directly to the GPU runner.

## Reuse and fallback

The GPU subprocess hook replaces a matching `load_inline` call with an import
of its compiled library. It checks source strings, compile options, build
environment, Python/PyTorch/CUDA ABI, and the hashes of compiler-reported header
dependencies. A packet also identifies the task sources and runner image.

| Submission or build condition | Behavior |
| --- | --- |
| Import-time `load_inline`, compatible inputs | Build on CPU and reuse on GPU |
| CUDA tensors or GPU queries during import | CPU import fails; run normally on GPU |
| Compilation inside `custom_kernel` | Compile normally on GPU when called |
| GPU-dependent source or header changes | Reject that artifact and compile on GPU |
| Custom linker flags (`extra_ldflags`) or non-Python libraries | Existing GPU path |
| Triton, CuTe, plain PyTorch, or Python with no detected `load_inline` | Existing GPU path |
| CPU service failure, timeout, missing function, oversized artifact | Existing GPU path |

The hook falls back per call, so a submission can reuse one extension and compile
another on the GPU. Failing correctness checks are still failures: fallback does
not retry evaluation or alter its result. Raw CUDA submissions are unchanged.

The CPU function uses the runner image, 4 CPU cores, 8 GiB RAM and a 120-second
import budget. Containers are single-use, Modal access is restricted, and the
PCH Volume is mounted read-only on both CPU and GPU. Artifact bundles are scoped
to the request, compressed, and limited to 1 MiB in transit / 64 MiB unpacked.
Larger results use the normal GPU path; this avoids Modal's large-result blob
upload from a restricted worker. There is no persistent binary cache.

`FullResult.cpu_compile` records the CPU duration, artifact count, reuse/fallback
counts and reason. It is diagnostic information, not a correctness signal.

## Operations

Deploy the runner before restarting the bot with the new launcher:

```bash
PYTHONPATH=src:src/runners uv run modal deploy src/runners/modal_runner_archs.py
```

`KERNELBOT_CPU_COMPILE=0` on the bot disables the CPU attempt. Set
`KERNELBOT_MODAL_APP` on both deployment and bot to use a separate runner app;
the default remains `discord-bot-runner`. A launcher talking to an older runner
falls back if the CPU function is unavailable.

## Local debug with real Modal workers

```bash
uv sync --extra dev
PYTHONPATH=src:src/runners uv run python scripts/debug_python_precompile.py --gpu T4
```

The debug command builds ordinary task configs and calls `ModalLauncher`. Its
function lookup uses the registered functions in a temporary Modal app, so it
does not deploy over the live runner or require a local database. CPU and GPU
execution use the configured real Modal account. Results are saved to
`/tmp/kernelbot-native-cpu-compile.json`.

The inline case uses the unchanged vector-add example and runs test, benchmark,
and leaderboard modes, with a 256-by-256 benchmark. Additional cases exercise
GPU work during import, lazy compilation, Triton, plain PyTorch, and source
generation and included headers that differ between CPU and GPU. Use
`--cases inline` to run only the first case.

Verified on a real Tesla T4: the unchanged inline submission passed all three
evaluation phases with six artifact loads and no GPU compilation fallback.
The changed-header case rejected the CPU library and passed after GPU compilation
([Modal run](https://modal.com/apps/coreauto/main/ap-b0aEIY861mwk1q0kVHvblV)).
GPU work during import, lazy compilation, Triton, plain PyTorch, and changed source
also passed through their expected paths
([compatibility run](https://modal.com/apps/coreauto/main/ap-JHjO3BkqWXZC39WN7WWKHs)).
The existing Modal integration tests assert reuse on T4 and H100.

Local tests:

```bash
uv run pytest tests/test_python_precompile.py tests/test_inline_artifacts.py tests/test_modal.py -m 'not integration'
```
