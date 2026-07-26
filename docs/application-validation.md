# Application validation

KernelBot can run a small, problem-owned workload against the current top 10
submissions. A problem opts in with two fields in `reference-kernels`:

```yaml
validation:
  version: cholesky-natural-gradient-v1
  script: validation.py
```

The version invalidates old results when the workload changes. KernelBot loads
the script beside the problem, adds it to the problem's normal source files,
and runs it on B200 through the existing Modal app.

The script receives the version in `KERNELBOT_VALIDATION_CONFIG` and prints one
JSON result:

```json
{
  "passed_shapes": 8,
  "total_shapes": 8,
  "fully_validated": true,
  "geomean_sync_wall_speedup": 1.42,
  "results": []
}
```

## Schedule

Every day at 22:00 `America/Los_Angeles`, one KernelBot replica claims each
`(leaderboard, GPU, contract version, local date)` sweep. It snapshots the
current best submission from each of the top 10 users and runs at most two
Modal jobs concurrently. The database claim prevents duplicate sweeps.

Set `APPLICATION_VALIDATION_ENABLED=false` to disable the scheduler. A failed
job is recorded as `VALIDATION ERROR`; a completed job is stored as `X/Y
VALIDATED`.

## Local debug

Modal already reads `MODAL_ENVIRONMENT`, so KernelBot does not need separate
environment plumbing:

```bash
modal environment create cholesky-validation-debug
modal deploy --env cholesky-validation-debug src/runners/modal_runner_archs.py

MODAL_ENVIRONMENT=cholesky-validation-debug \
APPLICATION_VALIDATION_ENABLED=false \
python src/kernelbot/main.py --api-only --debug
```

Run one sweep synchronously:

```bash
curl -X POST \
  -H "Authorization: Bearer ${ADMIN_TOKEN}" \
  "http://localhost:8000/admin/application-validations/cholesky/B200?wait=true"
```

Roll out KernelBot's migration and runner first, then the reference-kernels
contract, then the Kernelboard badge.
