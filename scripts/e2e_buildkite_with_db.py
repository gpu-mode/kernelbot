#!/usr/bin/env python3
"""End-to-end test for Buildkite integration with database storage.

This script:
1. Creates a test leaderboard in the local database
2. Submits a real kernel evaluation job to Buildkite
3. Stores results in the PostgreSQL database
4. Verifies everything is stored correctly

Usage:
    BUILDKITE_API_TOKEN=xxx uv run python scripts/e2e_buildkite_with_db.py

Options:
    --queue <name>      Buildkite queue (default: test)
    --org <slug>        Buildkite org (default: gpu-mode)
    --pipeline <slug>   Pipeline name (default: kernelbot)
    --example <name>    Example to run (default: vectoradd_py)
    --cleanup           Delete the test leaderboard after the test
    --dry-run           Print config without submitting
"""

import argparse
import asyncio
import datetime
import os
import sys
from pathlib import Path

# Add src to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class SimpleReporter:
    """Simple progress reporter for CLI output."""

    def __init__(self, title: str = ""):
        self.title = title
        self.messages = []

    async def push(self, msg):
        self.messages.append(msg)
        print(f"  [PUSH] {msg}")

    async def update(self, msg):
        print(f"  [UPDATE] {msg}")

    async def update_title(self, title):
        self.title = title
        print(f"  [TITLE] {title}")

    async def display_report(self, title, report):
        print(f"\n  [REPORT] {title}")
        for line in report:
            print(f"    {line}")


class MultiReporter:
    """Multi-run progress reporter."""

    def __init__(self):
        self.runs = []

    def add_run(self, name: str) -> SimpleReporter:
        reporter = SimpleReporter(name)
        self.runs.append(reporter)
        print(f"\n--- Run: {name} ---")
        return reporter

    async def show(self, msg):
        print(f"\n[SHOW] {msg}")


async def main():
    parser = argparse.ArgumentParser(description="E2E Buildkite test with database storage")
    parser.add_argument("--queue", default="test", help="Buildkite queue (default: test)")
    parser.add_argument("--org", default="gpu-mode", help="Buildkite org slug")
    parser.add_argument("--pipeline", default="kernelbot", help="Pipeline slug")
    parser.add_argument("--example", default="vectoradd_py", help="Example to run")
    parser.add_argument("--mode", choices=["test", "leaderboard"], default="test", help="Submission mode")
    parser.add_argument("--cleanup", action="store_true", help="Delete test leaderboard after test")
    parser.add_argument("--dry-run", action="store_true", help="Print config without submitting")
    args = parser.parse_args()

    # Check for required environment variables
    token = os.environ.get("BUILDKITE_API_TOKEN")
    if not token:
        print("ERROR: BUILDKITE_API_TOKEN environment variable not set")
        print("\nTo get a token:")
        print("  1. Go to https://buildkite.com/user/api-access-tokens")
        print("  2. Create token with: read_builds, write_builds, read_artifacts, read_agents")
        sys.exit(1)

    database_url = os.environ.get("DATABASE_URL", "postgresql://marksaroufim@localhost:5432/kernelbot")
    disable_ssl = os.environ.get("DISABLE_SSL", "true")

    print("=" * 60)
    print("Buildkite E2E Test with Database Storage")
    print("=" * 60)
    print(f"Organization: {args.org}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Queue: {args.queue}")
    print(f"Example: {args.example}")
    print(f"Mode: {args.mode}")
    print(f"Database: {database_url}")
    print()

    # Import kernelbot modules
    from libkernelbot.consts import BuildkiteGPU, SubmissionMode
    from libkernelbot.launchers.buildkite import BuildkiteConfig, BuildkiteLauncher
    from libkernelbot.leaderboard_db import LeaderboardDB
    from libkernelbot.task import make_task_definition

    # Set up database connection
    db = LeaderboardDB(url=database_url, ssl_mode="disable" if disable_ssl else "require")

    # Find example
    project_root = Path(__file__).parent.parent
    task_path = project_root / "examples" / args.example

    if not task_path.exists():
        print(f"ERROR: Example '{args.example}' not found at {task_path}")
        print("Available examples:")
        for p in (project_root / "examples").iterdir():
            if p.is_dir() and (p / "task.yml").exists():
                print(f"  - {p.name}")
        sys.exit(1)

    # Load task definition
    task_definition = make_task_definition(task_path)
    leaderboard_name = f"e2e-test-{args.example}"

    # Find submission file
    for name in ["submission_triton.py", "submission.py", "submission_cuda_inline.py"]:
        if (task_path / name).exists():
            submission_file = task_path / name
            break
    else:
        print(f"ERROR: No submission file found in {task_path}")
        sys.exit(1)

    submission_code = submission_file.read_text()

    print(f"Task: {task_path.name}")
    print(f"Submission: {submission_file.name}")
    print(f"Leaderboard: {leaderboard_name}")

    if args.dry_run:
        print("\n[DRY RUN] Would create leaderboard and submit job")
        config_keys = list(task_definition.task.config.keys()) if task_definition.task.config else "None"
        print(f"  Task config keys: {config_keys}")
        return

    # Step 1: Create test leaderboard
    print("\n" + "=" * 60)
    print("Step 1: Creating test leaderboard")
    print("=" * 60)

    with db:
        # Check if leaderboard already exists
        existing = db.get_leaderboard_names()
        if leaderboard_name in existing:
            print(f"  Leaderboard '{leaderboard_name}' already exists, deleting...")
            db.delete_leaderboard(leaderboard_name, force=True)

        # Create leaderboard
        deadline = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=30)
        lb_id = db.create_leaderboard(
            name=leaderboard_name,
            deadline=deadline,
            definition=task_definition,
            creator_id=1,  # Test user
            forum_id=0,
            gpu_types=["L40S_BK"],  # Buildkite test queue GPU
        )
        print(f"  Created leaderboard with ID: {lb_id}")

    # Step 2: Set up backend with Buildkite launcher
    print("\n" + "=" * 60)
    print("Step 2: Setting up Buildkite launcher")
    print("=" * 60)

    launcher = BuildkiteLauncher(
        BuildkiteConfig(
            org_slug=args.org,
            pipeline_slug=args.pipeline,
            api_token=token,
        )
    )

    # Check queue status
    queue_status = await launcher.get_queue_status(args.queue)
    print(f"  Queue: {queue_status.get('queue')}")
    print(f"  Total agents: {queue_status.get('total')}")
    print(f"  Idle agents: {queue_status.get('idle')}")
    for agent in queue_status.get("agents", []):
        print(f"    - {agent['name']}: {agent['state']} (busy={agent['busy']})")

    if queue_status.get("total", 0) == 0:
        print("\n  WARNING: No agents in queue. Job may wait indefinitely.")
        print("  Make sure you have agents running on the Buildkite queue.")

    # Step 3: Create submission and run evaluation
    print("\n" + "=" * 60)
    print("Step 3: Creating submission and running evaluation")
    print("=" * 60)

    with db:
        # Create submission entry
        submission_id = db.create_submission(
            leaderboard=leaderboard_name,
            file_name=submission_file.name,
            user_id=1,  # Test user
            code=submission_code,
            time=datetime.datetime.now(datetime.timezone.utc),
            user_name="e2e-test-user",
        )
        print(f"  Created submission with ID: {submission_id}")

    # Build task config
    from libkernelbot.task import build_task_config

    submission_mode = SubmissionMode.LEADERBOARD if args.mode == "leaderboard" else SubmissionMode.TEST
    config = build_task_config(
        task=task_definition.task,
        submission_content=submission_code,
        arch=0,  # Will be set by runner
        mode=submission_mode,
    )
    config["submission_id"] = submission_id

    # Run on Buildkite
    print("\n  Submitting to Buildkite...")
    gpu_type = BuildkiteGPU.L40S_BK
    reporter = SimpleReporter(f"Test run on {gpu_type.name}")

    result = await launcher.run_submission(config, gpu_type, reporter)

    print(f"\n  Result: success={result.success}")
    if result.error:
        print(f"  Error: {result.error}")
    print(f"  System: {result.system}")

    # Step 4: Store results in database
    print("\n" + "=" * 60)
    print("Step 4: Storing results in database")
    print("=" * 60)

    if result.success:
        with db:
            for run_name, run_result in result.runs.items():
                if run_result.run is None:
                    print(f"  Skipping {run_name}: no run result")
                    continue

                score = None
                if run_name == "leaderboard" and run_result.run.passed:
                    # Compute score for leaderboard runs
                    from libkernelbot.submission import compute_score
                    score = compute_score(result, task_definition.task, submission_id)

                db.create_submission_run(
                    submission=submission_id,
                    start=run_result.start,
                    end=run_result.end,
                    mode=run_name,
                    runner=gpu_type.name,
                    score=score,
                    secret=False,
                    compilation=run_result.compilation,
                    result=run_result.run,
                    system=result.system,
                )
                passed = run_result.run.passed
                duration = run_result.run.duration
                print(f"  Stored run: {run_name} (passed={passed}, duration={duration:.2f}s)")

            # Mark submission as done
            db.mark_submission_done(submission_id)
            print(f"\n  Marked submission {submission_id} as done")

    # Step 5: Verify data in database
    print("\n" + "=" * 60)
    print("Step 5: Verifying data in database")
    print("=" * 60)

    with db:
        submission = db.get_submission_by_id(submission_id)
        if submission:
            print(f"  Submission ID: {submission['submission_id']}")
            print(f"  Leaderboard: {submission['leaderboard_name']}")
            print(f"  File: {submission['file_name']}")
            print(f"  Done: {submission['done']}")
            print(f"  Runs: {len(submission['runs'])}")
            for run in submission['runs']:
                print(f"    - {run['mode']}: passed={run['passed']}, runner={run['runner']}")
                if run.get('system'):
                    gpu_name = run['system'].get('gpu', 'unknown') if isinstance(run['system'], dict) else 'unknown'
                    print(f"      GPU: {gpu_name}")
        else:
            print("  ERROR: Could not retrieve submission from database!")

    # Step 6: Show summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Leaderboard: {leaderboard_name}")
    print(f"  Submission ID: {submission_id}")
    print(f"  Success: {result.success}")
    if result.runs:
        for name, run in result.runs.items():
            if run.run:
                print(f"  {name}: passed={run.run.passed}, duration={run.run.duration:.2f}s")

    # Cleanup if requested
    if args.cleanup:
        print("\n" + "=" * 60)
        print("Cleanup")
        print("=" * 60)
        with db:
            db.delete_leaderboard(leaderboard_name, force=True)
            print(f"  Deleted leaderboard: {leaderboard_name}")

    print("\n" + "=" * 60)
    print("E2E Test Complete!")
    print("=" * 60)

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    asyncio.run(main())
