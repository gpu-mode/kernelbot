#!/usr/bin/env python3
"""End-to-end test for Buildkite integration.

Usage:
    BUILDKITE_API_TOKEN=xxx python tests/e2e_buildkite_test.py [--queue QUEUE]

This script:
1. Creates a simple test job
2. Submits it to Buildkite
3. Waits for completion
4. Prints the result
"""

import argparse
import asyncio
import os
import sys

# Add src to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


async def main():
    parser = argparse.ArgumentParser(description="E2E test for Buildkite integration")
    parser.add_argument("--queue", default="test", help="Buildkite queue name (default: test)")
    parser.add_argument("--org", default="gpu-mode", help="Buildkite org slug")
    parser.add_argument("--pipeline", default="kernelbot", help="Buildkite pipeline slug")
    parser.add_argument("--dry-run", action="store_true", help="Just print config, don't submit")
    args = parser.parse_args()

    token = os.environ.get("BUILDKITE_API_TOKEN")
    if not token:
        print("ERROR: BUILDKITE_API_TOKEN environment variable not set")
        sys.exit(1)

    from libkernelbot.launchers.buildkite import BuildkiteConfig, BuildkiteLauncher

    config = BuildkiteConfig(
        org_slug=args.org,
        pipeline_slug=args.pipeline,
        api_token=token,
    )

    print("=== Buildkite E2E Test ===")
    print(f"Organization: {config.org_slug}")
    print(f"Pipeline: {config.pipeline_slug}")
    print(f"Queue: {args.queue}")
    print()

    # Simple test config - just print GPU info
    test_config = {
        "lang": "py",
        "mode": "test",
        "sources": {
            "submission.py": """
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Device count: {torch.cuda.device_count()}")
""",
        },
        "main": "submission.py",
        "tests": [],
        "benchmarks": [],
    }

    if args.dry_run:
        print("Dry run - config would be:")
        import json
        print(json.dumps(test_config, indent=2))
        return

    launcher = BuildkiteLauncher(config)

    # Create a simple status reporter
    class SimpleReporter:
        async def push(self, msg):
            print(f"[STATUS] {msg}")

        async def update(self, msg):
            print(f"[UPDATE] {msg}")

    print("Submitting test job...")
    result = await launcher._launch(
        run_id="e2e-test",
        config=test_config,
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
        import json
        print(f"Result: {json.dumps(result.result, indent=2)}")

    # Also test queue status
    print()
    print("=== Queue Status ===")
    status = await launcher.get_queue_status(args.queue)
    print(f"Queue: {status.get('queue')}")
    print(f"Total agents: {status.get('total')}")
    print(f"Idle agents: {status.get('idle')}")
    for agent in status.get("agents", []):
        print(f"  - {agent['name']}: {agent['state']} (busy={agent['busy']})")


if __name__ == "__main__":
    asyncio.run(main())
