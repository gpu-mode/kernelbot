#!/usr/bin/env python3
"""
Basic smoke test for the Northflank launcher.

This script tests:
1. Launcher initialization with proper credentials
2. Job triggering with a simple payload
3. Status polling until completion
4. Result retrieval from logs

Usage:
    python scripts/test_northflank_basic.py

Environment variables required:
    NORTHFLANK_API_TOKEN - API token for Northflank
    NORTHFLANK_PROJECT_ID - Project ID
    NORTHFLANK_AMD_JOB_ID - AMD GPU job ID
    NORTHFLANK_NVIDIA_JOB_ID - (optional) NVIDIA GPU job ID
"""

import asyncio
import json
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from libkernelbot.consts import get_gpu_by_name
from libkernelbot.launchers.northflank import NorthflankLauncher
from libkernelbot.report import RunProgressReporter


class ConsoleProgressReporter(RunProgressReporter):
    """Simple console-based progress reporter for testing."""

    def __init__(self, title: str):
        super().__init__(title)
        print(f"[{self.title}]")

    async def _update_message(self):
        """Print the last line to console."""
        if self.lines:
            print(f"  {self.lines[-1]}")

    async def display_report(self, title: str, report):
        """Display report - not implemented for console."""
        print(f"[Report: {title}]")


async def main():
    """Run a basic smoke test of the Northflank launcher."""

    # Check for required environment variables
    required_vars = [
        "NORTHFLANK_API_TOKEN",
        "NORTHFLANK_PROJECT_ID",
        "NORTHFLANK_JOB_ID",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        print(f"❌ Error: Missing required environment variables: {', '.join(missing_vars)}")
        print("\nRequired environment variables:")
        print("  NORTHFLANK_API_TOKEN - Your Northflank API token")
        print("  NORTHFLANK_PROJECT_ID - Your Northflank project ID")
        print("  NORTHFLANK_JOB_ID - Job ID for GPU workloads")
        print("\nOptional:")
        print("  NORTHFLANK_REPO_URL - Git repository URL")
        print("  NORTHFLANK_REPO_BRANCH - Git branch to clone")
        return 1

    print("=" * 60)
    print("Northflank Launcher - Basic Smoke Test")
    print("=" * 60)

    # Initialize the launcher
    print("\n1️⃣  Initializing Northflank launcher...")
    try:
        launcher = NorthflankLauncher(
            api_token=os.environ["NORTHFLANK_API_TOKEN"],
            project_id=os.environ["NORTHFLANK_PROJECT_ID"],
            job_id=os.environ["NORTHFLANK_JOB_ID"],
            repo_url=os.environ.get("NORTHFLANK_REPO_URL"),
            repo_branch=os.environ.get("NORTHFLANK_REPO_BRANCH"),
        )
        print("✅ Launcher initialized successfully")
        print(f"   Project ID: {launcher.project_id}")
        print(f"   Job ID: {launcher.job_id}")
        print(f"   Repo: {launcher.repo_url}")
        print(f"   Branch: {launcher.repo_branch}")
    except Exception as e:
        print(f"❌ Failed to initialize launcher: {e}")
        return 1

    # Load test payload
    print("\n2️⃣  Loading test payload...")
    payload_path = Path(__file__).parent / "northflank_test_payload.json"
    if not payload_path.exists():
        print(f"❌ Test payload not found at: {payload_path}")
        return 1

    payload = json.loads(payload_path.read_text())
    print("✅ Test payload loaded")
    print(f"   Language: {payload['lang']}")
    print(f"   Mode: {payload['mode']}")

    # Prepare config for submission
    config = {
        **payload,
        "mode": "test",
        "test_timeout": 180,
        "problem": "smoke_test",
    }

    # Select GPU
    print("\n3️⃣  Selecting GPU...")
    gpu = get_gpu_by_name("MI300")
    if not gpu:
        print("❌ Failed to get GPU type MI300")
        return 1
    print(f"✅ Selected GPU: {gpu.name} ({gpu.value})")

    # Create status reporter
    status = ConsoleProgressReporter(title="Smoke Test")

    # Run submission
    print("\n4️⃣  Triggering Northflank job...")
    print("   (This will take a few minutes...)")

    try:
        result = await launcher.run_submission(config, gpu, status)
    except Exception as e:
        print(f"\n❌ Job execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Check results
    print("\n5️⃣  Checking results...")
    print(f"   Success: {result.success}")
    print(f"   Error: {result.error}")
    print(f"   Runs: {len(result.runs)}")

    if result.system:
        print(f"   System Info:")
        print(f"     Platform: {getattr(result.system, 'platform', 'N/A')}")
        print(f"     GPU: {getattr(result.system, 'gpu_name', 'N/A')}")

    if result.runs:
        print(f"\n   Run details:")
        for run_name, run_result in result.runs.items():
            print(f"     {run_name}:")
            print(f"       Start: {run_result.start}")
            print(f"       End: {run_result.end}")
            if run_result.run:
                print(f"       Exit code: {run_result.run.exit_code}")
                print(f"       Success: {run_result.run.success}")
                if run_result.run.stdout:
                    print(f"       Stdout (first 200 chars): {run_result.run.stdout[:200]}")
                if run_result.run.stderr:
                    print(f"       Stderr: {run_result.run.stderr[:200]}")

    if result.success:
        print("\n✅ Smoke test PASSED!")
        print("   The Northflank launcher is working correctly.")
        return 0
    else:
        print("\n❌ Smoke test FAILED!")
        print(f"   Error: {result.error}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
