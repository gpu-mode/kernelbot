# Model Competitions — Reused Components

This document lists every component reused without modification when running
e2e model competitions. Use this as a reference for what **not** to change.

## User Management & Auth

| File | Component | Notes |
|------|-----------|-------|
| `src/libkernelbot/leaderboard_db.py` | `validate_identity()`, `validate_cli_id()`, `init_user_from_cli()`, `create_user_from_cli()` | Same auth flow for CLI and web users |
| `src/kernelbot/api/main.py` | `validate_cli_header()`, `validate_user_header()` | FastAPI dependency injection for auth headers |
| `src/libkernelbot/db_types.py` | `IdentityType` enum | CLI / WEB / UNKNOWN identity types |

## Database Tables (no migrations needed)

| Table | Purpose |
|-------|---------|
| `leaderboard.user_info` | User identity and CLI/web auth tokens |
| `leaderboard.submission` | Submission records — same columns, `code_id` references tarball bytes |
| `leaderboard.runs` | Per-GPU run results — `result` JSONB stores model metrics instead of kernel timings |
| `leaderboard.code_files` | Content-addressable storage — BYTEA column stores tarball bytes |
| `leaderboard.submission_job_status` | Async job lifecycle tracking with heartbeats |
| `leaderboard.leaderboard` | Leaderboard definitions — `task` JSONB stores `ModelTaskData` |
| `leaderboard.gpu_type` | GPU types per leaderboard |
| `leaderboard.templates` | Not used for model competitions but schema unchanged |

## Backend Orchestration

| File | Component | Notes |
|------|-----------|-------|
| `src/libkernelbot/backend.py` | `KernelBackend.submit_full()` | Fan-out to GPUs, secret runs, `asyncio.gather`, `mark_submission_done` — identical flow |
| `src/libkernelbot/backend.py` | `KernelBackend.submit_leaderboard()` | Score computation dispatch, `create_submission_run` DB writes — reused with extended scoring |
| `src/libkernelbot/backend.py` | `KernelBackend.register_launcher()`, `launcher_map` | Strategy pattern dispatch by GPU type — unchanged |

## Job Management

| File | Component | Notes |
|------|-----------|-------|
| `src/libkernelbot/background_submission_manager.py` | `BackgroundSubmissionManager` | Async queue, worker pool, heartbeat loop, auto-scaling (2-24 workers) — all reused |
| `src/libkernelbot/leaderboard_db.py` | `upsert_submission_job_status()`, `update_heartbeat_if_active()` | Job status tracking — unchanged |

## Launcher Infrastructure

| File | Component | Notes |
|------|-----------|-------|
| `src/libkernelbot/launchers/launcher.py` | `Launcher` base class | Abstract interface — unchanged |
| `src/libkernelbot/launchers/modal.py` | `ModalLauncher` class structure | `run_submission()` method reused — only function name resolution extended |
| `src/runners/modal_runner.py` | `modal_run_config()`, `timeout()` context manager | Same entry point wrapping `run_config()` |

## API Endpoints

| File | Component | Notes |
|------|-----------|-------|
| `src/kernelbot/api/main.py` | `POST /submission/{lb}/{gpu}/{mode}` | Same endpoint shape — validation logic branched by lang type |
| `src/kernelbot/api/main.py` | SSE streaming response format | `event: status`, `event: result`, `event: error` — unchanged |
| `src/kernelbot/api/main.py` | Rate limiting, `_submit_limiter` | Same global rate limiter |

## Progress Reporting

| File | Component | Notes |
|------|-----------|-------|
| `src/libkernelbot/report.py` | `MultiProgressReporter`, `RunProgressReporter` | Status update streaming — unchanged |

## Leaderboard Management

| File | Component | Notes |
|------|-----------|-------|
| `src/libkernelbot/leaderboard_db.py` | `create_leaderboard()`, `update_leaderboard()`, `delete_leaderboard()` | CRUD operations — unchanged, `task` JSONB accepts any task format |
| `src/libkernelbot/leaderboard_db.py` | `get_leaderboard()`, `get_leaderboards()`, `get_leaderboard_names()` | Query operations — unchanged |
| `src/libkernelbot/problem_sync.py` | `sync_problems()`, `create_update_plan()` | Problem sync from reference-kernels repo — works with model `task.yml` files |

## Anti-Cheat

| Component | Kernel Competitions | Model Competitions |
|-----------|--------------------|--------------------|
| Secret seed mechanism | `check_implementation` with secret inputs | Perplexity check against baseline |
| `leaderboard.secret_seed` column | Used | Available (perplexity eval uses fixed dataset) |
| Secret runs (`SubmissionMode.PRIVATE`) | Dual public+private runs | Same dual-run pattern |

## Data Types & Result Format

| File | Component | Notes |
|------|-----------|-------|
| `src/libkernelbot/run_eval.py` | `FullResult`, `RunResult`, `EvalResult`, `CompileResult`, `SystemInfo` | Same dataclasses — `result` dict stores different keys for model metrics |
| `src/libkernelbot/db_types.py` | `LeaderboardItem`, `SubmissionItem`, `RunItem`, `LeaderboardRankedEntry` | Same TypedDicts — score semantics extended with direction |
