#!/usr/bin/env python3
"""Submit a test job to Buildkite and download the result.

Usage:
    # Simple test (just writes dummy result.json):
    BUILDKITE_API_TOKEN=xxx python scripts/submit_buildkite_job.py

    # Real evaluation with vectoradd example:
    BUILDKITE_API_TOKEN=xxx python scripts/submit_buildkite_job.py --eval vectoradd_py

    # Real evaluation with identity example:
    BUILDKITE_API_TOKEN=xxx python scripts/submit_buildkite_job.py --eval identity_py
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from libkernelbot.consts import BuildkiteGPU, SubmissionMode
from libkernelbot.launchers.buildkite import BuildkiteConfig, BuildkiteLauncher
from libkernelbot.task import build_task_config, make_task_definition


class SimpleReporter:
    async def push(self, msg):
        print(f"[STATUS] {msg}")

    async def update(self, msg):
        print(f"[UPDATE] {msg}")


async def main():
    parser = argparse.ArgumentParser(description="Submit a test job to Buildkite")
    parser.add_argument("--org", default="mark-saroufim", help="Buildkite org slug")
    parser.add_argument("--pipeline", default="kernelbot", help="Pipeline slug")
    parser.add_argument("--queue", default="test", help="Queue name")
    parser.add_argument("--run-id", default="manual-test", help="Run ID for this job")
    parser.add_argument(
        "--eval",
        type=str,
        default=None,
        help="Run real evaluation with example (e.g., 'vectoradd_py', 'identity_py')",
    )
    parser.add_argument(
        "--submission",
        type=str,
        default=None,
        help="Submission file to use (default: auto-detect)",
    )
    args = parser.parse_args()

    token = os.environ.get("BUILDKITE_API_TOKEN")
    if not token:
        print("ERROR: Set BUILDKITE_API_TOKEN environment variable")
        sys.exit(1)

    print("=== Buildkite Job Submission ===")
    print(f"Org: {args.org}")
    print(f"Pipeline: {args.pipeline}")
    print(f"Queue: {args.queue}")
    print(f"Run ID: {args.run_id}")

    launcher = BuildkiteLauncher(
        BuildkiteConfig(
            org_slug=args.org,
            pipeline_slug=args.pipeline,
            api_token=token,
        )
    )

    if args.eval:
        # Real evaluation mode
        print(f"Eval: {args.eval}")
        print()

        project_root = Path(__file__).parent.parent
        task_path = project_root / "examples" / args.eval

        if not task_path.exists():
            print(f"ERROR: Example '{args.eval}' not found at {task_path}")
            print("Available examples:")
            for p in (project_root / "examples").iterdir():
                if p.is_dir() and (p / "task.yml").exists():
                    print(f"  - {p.name}")
            sys.exit(1)

        task_definition = make_task_definition(task_path)

        # Find submission file
        if args.submission:
            submission_file = task_path / args.submission
        else:
            # Try common submission names
            for name in ["submission_triton.py", "submission.py", "submission_cuda_inline.py"]:
                if (task_path / name).exists():
                    submission_file = task_path / name
                    break
            else:
                print(f"ERROR: No submission file found in {task_path}")
                sys.exit(1)

        print(f"Task: {task_path.name}")
        print(f"Submission: {submission_file.name}")

        submission_content = submission_file.read_text()

        config = build_task_config(
            task=task_definition.task,
            submission_content=submission_content,
            arch=0,
            mode=SubmissionMode.TEST,
        )

        gpu_type = BuildkiteGPU.L40S_BK
        result = await launcher.run_submission(config, gpu_type, SimpleReporter())

        print()
        print("=== Result ===")
        print(f"Success: {result.success}")
        if result.error:
            print(f"Error: {result.error}")
        print(f"System: {result.system}")
        if result.runs:
            for name, run in result.runs.items():
                print(f"\n{name}:")
                print(f"  Passed: {run.run.passed if run.run else 'N/A'}")
                print(f"  Duration: {run.run.duration if run.run else 'N/A'}s")
                if run.run and run.run.result:
                    print(f"  Result: {run.run.result}")

    else:
        # Simple test mode
        print("Mode: Simple test (no evaluation)")
        print()

        config = {
            "test": True,
            "message": "Hello from manual test",
            "run_id": args.run_id,
        }

        print("Submitting job...")
        result = await launcher._launch(
            run_id=args.run_id,
            config=config,
            queue=args.queue,
            status=SimpleReporter(),
        )

        print()
        print("=== Result ===")
        print(f"Success: {result.success}")
        if result.error:
            print(f"Error: {result.error}")
        if result.build_url:
            print(f"Build URL: {result.build_url}")
        if result.result:
            print("Downloaded artifact:")
            print(json.dumps(result.result, indent=2))
        else:
            print("No artifact downloaded")

    sys.exit(0 if (result.success if hasattr(result, "success") else True) else 1)


if __name__ == "__main__":
    asyncio.run(main())
