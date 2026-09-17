# Hosted Nsight Compute profiling

`popcorn submit submission.py --leaderboard qr_v2 --profile --benchmark-index 0`
uses the normal Popcorn identity and API. Users do not install Modal, create a
provider account, or supply provider credentials. KernelBot dispatches the job
to its GPU runner and streams back reports. `--profile-brev` remains an explicit,
separate CLI choice; failures never switch providers.

## API and capture contract

`POST /profile/{leaderboard_name}/{gpu_type}` accepts the usual
`X-Popcorn-Cli-Id` header and a multipart `file`, plus optional fields:

| Field | Meaning |
| --- | --- |
| `benchmark_index` | Zero-based index in the task's `benchmarks`; omit for all entries. |
| `ncu_kernel_name` | NCU kernel filter, including `regex:` expressions. |
| `ncu_kernel_name_base` | `function`, `demangled`, or `mangled`. |
| `ncu_launch_count` | Positive capture limit per benchmark; default 10. |

The route retains normal submission permissions, limits, and leaderboard GPU
validation. It supports the configured single-GPU NVIDIA Modal runners. Invalid
options and unsupported GPUs are rejected; unsupported evaluators fail with NCU
diagnostics. The dedicated route prevents older API versions from silently
ignoring capture options.

The runner sets `POPCORN_NCU=1`, follows child processes, and captures the NVTX
push/pop range `custom_kernel/`. It collects the full NCU section set with kernel
replay and leaves GPU clocks unchanged. Each benchmark returns a zip containing
`profile.ncu-rep`, `ncu-details.txt`, and `ncu-details.csv`. A missing capture is a
failure, even when the profiler process exits successfully.

`FullResult.profile_metadata` records selected benchmark specifications, capture
options, and a SHA-256 fingerprint of source/configuration fields. This is not a
git commit. Record the synced reference-kernels revision in operator validation
records; the client cannot update the hosted task through a local checkout.

## Rollout

Deploy the updated GPU runners and API before releasing the companion
[popcorn-cli change](https://github.com/gpu-mode/popcorn-cli/pull/80). The runner
image pins NCU 2025.2.1 and checks `ncu --version` during its build. NCU 2026.2
from CUDA 13.3 produced many NaN counters on the tested Modal B200; repeating
the same capture with 2025.2.1 restored hardware counters. Use the existing Modal deployment
workflow and API deployment process; no additional container privileges or
user-facing provider setup is introduced.

Sync an NCU-compatible task through the existing problem-update workflow, then
validate one benchmark through the public API using an ordinary Popcorn user.
Inspect the returned report and counter exports. The
[reference-kernels guide](https://github.com/gpu-mode/reference-kernels/pull/171)
describes the evaluator contract for new problems. Merely accepting `profile`
and running `torch.profiler` does not satisfy that contract.

## Validation

CPU tests cover API authentication and options, benchmark selection, capture
commands, exports, and empty captures. The GPU fixture is
`problems/linalg/qr_v2`, benchmark 0 (`batch=20, n=32, cond=1, seed=43214`), at
reference-kernels `51e22db671d36c1c76091c43c36a44546ba324a1`.
Only this problem/shape is used for the GPU integration check; other tasks and
GPU types require their own validation.

The integration check used the real FastAPI route, submission preparation,
KernelBackend, and GPU `run_config`, with a fake database and an ephemeral
launcher instead of production infrastructure. The CLI used normal Popcorn
header authentication with an empty `PATH` and no `MODAL_*` environment values.
It successfully saved and extracted a 7,593,257-byte report and both detail
exports. NCU reopened the report to produce those exports.

NCU 2025.2.1 collected 39 passes on NVIDIA B200 with PyTorch 2.12.0+cu130,
CUDA 13.3, and `regex:geqr2` / demangled / launch count 1. Observed counters:
322.78 us duration, 12.51% achieved occupancy, 0.42% SM throughput, 70.09% L2
hit rate, and 1,565,428 executed instructions. Six `ctc__*` metrics remained
unavailable. This validates the isolated implementation; the production API
has not been deployed as part of this change.

[Ephemeral validation run](https://modal.com/apps/coreauto/main/ap-p8eodYfHkZAMp0ZT5btR1y)
(access requires the operator's workspace permissions).
