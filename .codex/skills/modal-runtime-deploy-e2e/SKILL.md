---
name: modal-runtime-deploy-e2e
description: Upgrade shared Modal runtime dependencies in kernelbot and verify them end to end. Use when changing torch/CUDA or other shared Modal image dependencies, deploying the Modal app, and validating with both Modal integration tests and real popcorn leaderboard submissions.
---

# Modal Runtime Deploy and E2E

Use this when changing shared Modal dependencies in `kernelbot`, especially torch/CUDA, and when you need to prove the live leaderboard is actually using the new runtime.

## Scope

- Shared Modal image: `src/runners/modal_runner.py`
- GPU-bound Modal functions: `src/runners/modal_runner_archs.py`
- Live app name: `discord-bot-runner`
- Popcorn e2e path: generate invite if needed, join closed leaderboard, submit with `popcorn-cli`

## Workflow

1. Make the smallest dependency change in `src/runners/modal_runner.py`.
2. If changing torch/CUDA, inspect all later `.uv_pip_install(...)` blocks for conflicting CUDA/NCCL packages.
3. Deploy to Modal `pytest` first.
4. Run the narrow Modal integration test:

```bash
cd /Users/mark/Dev/kernelbot
env MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... \
  uv run --extra dev python -m pytest -s tests/test_modal.py -k 'test_modal_launcher_python_script and T4'
```

5. If that passes, deploy to Modal `main`:

```bash
cd /Users/mark/Dev/kernelbot/src/runners
env MODAL_TOKEN_ID=... MODAL_TOKEN_SECRET=... \
  /Users/mark/Dev/kernelbot/.venv/bin/modal deploy --env main modal_runner_archs.py
```

6. Run a real `popcorn` submission in `test` mode against the target leaderboard.
7. Confirm the returned report shows the expected `Torch:` version.
8. Only then run `--mode leaderboard` if the user asked for a ranked submission.

## Closed Leaderboards

Generate an invite with admin token:

```bash
cd /Users/mark/Dev/popcorn-cli
env POPCORN_API_URL=... POPCORN_ADMIN_TOKEN=... \
  cargo run --quiet -- admin generate-invites --leaderboards <leaderboard> --count 1
```

Join with the existing CLI identity in `~/.popcorn.yaml`:

```bash
cd /Users/mark/Dev/popcorn-cli
env POPCORN_API_URL=... \
  cargo run --quiet -- join '<invite_code>'
```

## Real E2E Submit

```bash
cd /Users/mark/Dev/popcorn-cli
env POPCORN_API_URL=... \
  cargo run --quiet -- submit --no-tui --leaderboard <leaderboard> --gpu A100 --mode test <submission.py>
```

Ranked submit:

```bash
cd /Users/mark/Dev/popcorn-cli
env POPCORN_API_URL=... \
  cargo run --quiet -- submit --no-tui --leaderboard <leaderboard> --gpu A100 --mode leaderboard <submission.py>
```

Check recent runs:

```bash
cd /Users/mark/Dev/popcorn-cli
env POPCORN_API_URL=... \
  cargo run --quiet -- submissions list --leaderboard <leaderboard> --limit 5
```

## Failure Mode To Remember

If a Modal run fails with:

```text
libtorch_cuda.so: undefined symbol: ncclDevCommCreate
```

then a later package install likely replaced torch's expected CUDA/NCCL dependency set. The practical fix is to install `torch` last so its dependency versions win.
