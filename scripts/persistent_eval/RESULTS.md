# Inline CUDA: persistence saves 19–28% of evaluation wall time

Measured on 2026-09-04 using KernelBot's bundled inline-CUDA FP16 vector-add
example, three benchmark shapes (1024², 2048², 4096²), and cold private build
caches for every submission. GPU measurements remain sequential.

## Turnaround

Each batch contains four submissions and includes worker startup/teardown.
Compilation and correctness checks are included; Modal image/container startup
and the once-per-experiment machine probe are excluded.

| Experiment | Fresh processes | Persistent worker | Less wall time |
|---|---:|---:|---:|
| Initial allocation, median of two counterbalanced rounds | 173.8 s | 125.4 s | 27.8% |
| Complete repeat on another allocation, one fresh/persistent round | 151.1 s | 122.2 s | 19.1% |

That is approximately **29–48 seconds saved per four-submission batch** on this
workload. Each comparison uses the same assigned GPU and CPU resources for both
modes. Absolute times vary across the two allocations; do not pool those times
into a single baseline. This small experiment is not a production workload
estimate or a confidence interval.

## What still takes time

Per-request phase medians from the complete repeat:

| Phase | Fresh processes | Persistent worker |
|---|---:|---:|
| Compile and import submission | 31.74 s | 29.85 s |
| Benchmark request, including validation and setup | 5.89 s | 0.41 s |

Keeping Python/CUDA alive removes most benchmark-request startup. Rebuilding
C++/CUDA dominates the remaining time. Every inline request produced its own
native library; persistent requests used `add_cuda.so`, then `add_cuda_v1.so`,
`add_cuda_v2.so`, and `add_cuda_v3.so` in separate cold build directories.
The phase medians are individual-request statistics, not additive batch totals.

## Kernel scores are not yet equivalent

The same correctness and CUDA-event timing function bodies produced different
reported kernel times. Below are medians of each submission's reported mean in
the complete repeat. These are GPU microseconds, distinct from evaluation wall
seconds above.

| Shape | Fresh, µs | Persistent, µs | Persistent change |
|---|---:|---:|---:|
| 1024² | 36.02 | 40.28 | +11.8% |
| 2048² | 108.37 | 123.65 | +14.1% |
| 4096² | 407.96 | 468.47 | +14.8% |

These 12–15% shifts require investigation before using the prototype for ranked
scores. This one ordered repeat does not separate process-state effects from
clock/thermal drift or measurement variability; GPU clocks were not pinned.
There is no claim that the kernel itself became faster or that the two execution
modes are score-equivalent.

## Correctness and code replacement

All 24 timed submissions passed across the two experiments. The complete repeat
also ran **correct addition → incorrect subtraction → correct addition** in one
persistent worker, retaining the requested extension name `add_cuda`. Observed
results were **pass → fail → pass**. The incorrect variant compiled successfully
and failed with numerical mismatches on all three shapes; this was not a compiler
or import failure. Restored code produced correct results again.

The worker still does not isolate arbitrary user code, unload native modules,
implement production request timeouts, or recover from a poisoned CUDA context.
This PR remains an experiment with no production runner changes.

## Evidence and reproduction

- [Initial timing observations](results/inline-initial-timings.json): two complete
  rounds, recovered from provider logs after a local DNS/connection failure. The
  final reload check was interrupted. Full phase/kernel records were not exported,
  so this file explicitly marks its limited detail.
- [Complete repeat](results/inline-repeat.json): all per-submission/phase timings,
  correctness output, kernel statistics, native-library names/hashes, and a hash
  of the full local result. Compiler logs are omitted from this compact artifact.
- [Repeat manifest](results/inline-repeat-manifest.json): runtime/source hashes,
  pinned image, resources, and stopped sandbox metadata. The uploaded source
  hashes match the PR runtime files at commit `4de31428`.
- [Run instructions](README.md): reproducible fresh/persistent comparison and
  correct/incorrect/correct check. The default runs two counterbalanced rounds;
  this complete repeat used `--rounds 1` after the original two-round matrix.

The complete repeat used one Tesla T4, eight reserved CPU cores, 32 GiB RAM,
CUDA 12.8.1 compiler image, Torch 2.7.1, and `MAX_JOBS=2`. It took 406 seconds
including local launch/admission overhead, with an estimated $0.067 GPU-only
cost at $0.000164/T4-second; CPU/RAM are excluded. Both experiment sandboxes are
stopped. No production deployment, submissions, or database writes were made.

Validation: repository-wide Ruff and three CPU regression tests pass. The
summarizer additionally audits the actual GPU result, requiring valid timing
statistics, successful compilation, native-library artifacts, and numerical
mismatch evidence for the negative reload test.
