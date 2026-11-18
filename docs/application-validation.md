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

## Manual batches

Application validation never runs automatically. An admin explicitly starts a
batch for one leaderboard and GPU. By default it snapshots the current best
submission from each of the top 10 users and runs at most two Modal jobs
concurrently. A failed job is recorded as `VALIDATION ERROR`; a completed job
is stored as `X/Y VALIDATED`.

## Local debug

Modal already reads `MODAL_ENVIRONMENT`, so KernelBot does not need separate
environment plumbing:

```bash
modal environment create cholesky-validation-debug
modal deploy --env cholesky-validation-debug src/runners/modal_runner_archs.py

MODAL_ENVIRONMENT=cholesky-validation-debug \
python src/kernelbot/main.py --api-only --debug
```

Run one sweep synchronously:

```bash
kernelbot-admin --api-url http://localhost:8000 \
  validate-top10 cholesky B200 --wait
```

Omit `--wait` to enqueue the sweep and return immediately. The command reads
`ADMIN_TOKEN` and, unless `--api-url` is supplied,
`DISCORD_CLUSTER_MANAGER_API_BASE_URL` from the environment.

For a one-time backfill of every ranked user's current best submission, add
`--all-users`:

```bash
kernelbot-admin validate-top10 cholesky B200 --all-users --only-missing
```

`--only-missing` skips exact submissions that already have a result for the
active validation contract, making it suitable for continuing a backfill after
leaderboard changes.

Roll out KernelBot's migration and runner first, then the reference-kernels
contract, then the Kernelboard badge.
