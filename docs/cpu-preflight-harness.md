# CPU preparation and GPU evaluation are different passes

**CPU compilation does not imply that every benchmark shape has been prepared.**
Reliable traversal of supported test cases needs a compile-only mode owned by
the evaluation harness. This document describes that limitation and the proposed
mode; it does not add a Triton/CuTe compilation stage to production dispatch.

## What exists today

[`ModalLauncher.run_submission`](../src/libkernelbot/launchers/modal.py) can run
CPU preparation before reserving the GPU. The current
[`python_precompile.py`](../src/libkernelbot/python_precompile.py) implementation
selects Python submissions containing `load_inline` and executes `submission.py`
on CPU. Compatible extensions compiled during that import can be reused later.
It does not call `custom_kernel` for every test or benchmark, and it does not
precompile Triton or CuTe specializations. Builds reached only inside
`custom_kernel` still happen on GPU.

## Why running the normal evaluator with fake tensors is insufficient

A typical correctness loop looks like this:

```python
for case in cases:
    data = generate_input(**case)
    output = custom_kernel(data)
    assert_close(output, reference(data))
```

Fake tensors describe metadata such as shape, dtype, and strides; they do not
contain the values needed to evaluate that numerical assertion. Even with a
launch interceptor and fake-compatible inputs, a CPU trace can stop at the
first correctness check, leaving later shapes undiscovered. GPU-dependent input
generation or initialization can stop it before any kernel is reached.

The stock [`examples/eval.py`](../examples/eval.py) has these dependencies:
`_run_single_test` generates real inputs, synchronizes CUDA, launches the
submission, and checks its result. `_run_single_benchmark` also checks
correctness before starting timing and can recheck newly seeded inputs.
Those operations belong in GPU evaluation. Fake execution cannot produce valid
benchmark timings or a correctness result.

Therefore, knowing that a task has ten shapes is not enough to promise that a
generic trace precompiles all ten. Moving a numerical assertion out of the CPU
preparation pass is a deliberate harness design decision; suppressing arbitrary
assertions while pretending to run the normal evaluator is not a substitute.

## Proposed compile-only mode

Use two passes over shared task definitions, rather than two independently
maintained evaluators or shape lists:

| Pass | Inputs and work | Outputs |
| --- | --- | --- |
| CPU preparation | Construct fake inputs from validated task metadata and invoke the unchanged kernel for each supported case; intercept launches to compile | Compatible artifacts and discovery diagnostics |
| GPU evaluation | Generate real inputs and run the existing checks, warmup, timing, and seeded rechecks | Correctness results and benchmark scores |

Only the GPU pass can determine correctness or scores. CPU preparation must
neither time fake execution nor fabricate tensor values to force a branch.

[`build_task_config`](../src/libkernelbot/task.py) already packages the task's
sources, tests, benchmarks, architecture, mode, and seed. A future preparation
mode should use the cases required by that actual evaluation, including both
tests and benchmarks when applicable. For example,
[`vectoradd_py/task.yml`](../examples/vectoradd_py/task.yml) has five test sizes
and five benchmark sizes. Preparing only its tests omits the five benchmark
inputs. Ten inputs can also share compiler specializations; the number of
shapes is not necessarily the number of compiled binaries.

The case definitions can be shared, but input construction still needs care.
[`vectoradd_py/reference.py`](../examples/vectoradd_py/reference.py) explicitly
constructs a CUDA `torch.Generator`. It cannot simply be assumed to work on a
CPU worker under fake-tensor mode. A task-owned adapter can describe its two
contiguous FP16 matrices. Other tasks may need strides, storage offsets, scalar
parameters, dynamic dimensions, or custom descriptors. Those contracts must be
validated against the real generator to prevent metadata drift.

This requires changes to the harness and supported task adapters, while keeping
users' `custom_kernel(data)` implementations and submission interface unchanged.
It does not require users to upload binaries or write their own build scripts.
The shared compile-only mode is **future work**, not current behavior.

## Submission flow and fallback

The proposed flow is:

```text
normal upload and validation → background submission job
  → bounded CPU preparation for supported cases
  → compatible artifact transfer
  → GPU allocation and the existing full evaluator
  → normal results and scores
```

CPU preparation must finish before the GPU is reserved. Asking a CPU compiler
for work while a GPU worker waits still occupies the GPU. Keep build artifacts
scoped to the submission, target, toolchain, source dependencies, and actual
public/private workload; do not expose private case metadata through public
artifacts or assume two differently seeded workloads have identical coverage.

Even a separate preparation mode cannot discover every path through arbitrary
submitted Python. Tensor-value-dependent branches, runtime GPU queries,
autotuning decisions, CUDA graphs, and unsupported descriptors can require GPU
execution. Missing or incompatible specializations should retain normal GPU
JIT behavior in a compatibility-oriented rollout. An evaluation failure must
remain a failure; fallback must not suppress or retry away correctness errors.

## Evidence needed before making this a default

- Verify that fake and real inputs agree on compilation-relevant metadata for
  every supported case, including benchmark-only shapes.
- Test partial discovery, missing artifacts, changed source/options/targets,
  and a case checked immediately after its launch. All real checks must still run.
- Verify cache hits and misses on the real GPU. Do not infer reuse merely from
  successful CPU compilation or a passing submission.
- Compare complete GPU allocation time, including transfer and setup, as well
  as total submission latency. CPU work and transfer can cost more than they save.
- Compare benchmark results through the original launch path, especially for
  short kernels where a different callable wrapper can change measured timings.

Keep expanded preflight opt-in until those properties are established for the
supported tasks. A successful compile-only experiment is not a guarantee that
most kernels benefit or that every shape can be prepared without a GPU.
