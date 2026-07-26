# Application validation

KernelBot can re-run the current top submissions inside a small,
problem-owned application workload. This complements the leaderboard's
operator correctness checks with an end-to-end signal such as training
convergence.

## Contract

The validation contract lives beside the problem in `reference-kernels`.
KernelBot stores the resolved contract in the leaderboard task JSON, so every
result is tied to an explicit version.

```yaml
validation:
  name: natural-gradient-training
  version: cholesky-natural-gradient-v1
  main: validation.py
  files:
    - {name: submission.py, source: "@SUBMISSION@"}
    - {name: validation.py, source: validation.py}
  timeout: 900
  top_k: 10
  max_concurrency: 2
  schedule:
    hour: 22
    minute: 0
    timezone: America/Los_Angeles
  settings:
    require_no_torch_fallback: true
    min_speedup: 1.0
  shapes:
    - {batch: 4096, n: 32, steps: 12}
```

The problem entrypoint receives `name`, `version`, `settings`, and `shapes` as
JSON in `KERNELBOT_VALIDATION_CONFIG`. It must print one JSON object as its last
stdout line. Aggregate fields are:

```json
{
  "passed_shapes": 8,
  "total_shapes": 8,
  "fully_validated": true,
  "geomean_sync_wall_speedup": 1.42,
  "results": []
}
```

`fully_validated` is accepted only when the result shape count matches the
versioned contract and every shape passes. Changing the workload or a gate
requires a new contract version.

## Nightly flow

At the contract's local scheduled time, each KernelBot replica tries to claim
one `(leaderboard, GPU, contract version, local date)` sweep. The database
unique constraint lets exactly one replica proceed.

The owner:

1. snapshots the current best submission for each of the top `top_k` users;
2. runs KernelGuard on every selected source;
3. launches at most `max_concurrency` isolated Modal jobs;
4. stores a versioned summary for each exact submission ID; and
5. marks the sweep complete.

Raw submitted source, stdout, and stderr are never stored in validation rows or
returned by the admin endpoint. Only whitelisted aggregate and per-shape
metrics are persisted.

The scheduler is enabled by default when a task has a validation contract.
Set `APPLICATION_VALIDATION_ENABLED=false` for an emergency stop. Polling
defaults to 60 seconds and can be changed with
`APPLICATION_VALIDATION_POLL_SECONDS`.

Application validation fails closed when `KERNELGUARD_ENABLED` is not enabled
or KernelGuard is unavailable.

## Local debug

Deploy the Modal functions into a non-production environment:

```bash
modal environment create cholesky-validation-debug
modal deploy --env cholesky-validation-debug src/runners/modal_runner_archs.py
```

Run a local API instance against a migrated development database:

```bash
MODAL_ENVIRONMENT=cholesky-validation-debug \
APPLICATION_VALIDATION_ENABLED=false \
KERNELGUARD_ENABLED=1 \
python src/kernelbot/main.py --api-only --debug
```

Trigger one synchronous sweep:

```bash
curl -X POST \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  "http://localhost:8000/admin/application-validations/cholesky/B200?wait=true"
```

Omit `wait=true` to enqueue the manual sweep and return immediately.

The rollout order is KernelBot migration and runner support, then the
`reference-kernels` contract, then the Kernelboard badge. Kernelboard only
shows results whose contract version matches the leaderboard's current task.
