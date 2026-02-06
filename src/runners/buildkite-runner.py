#!/usr/bin/env python3
"""Buildkite job runner for kernel evaluation."""

import base64
import json
import os
import sys
import zlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path


def serialize(obj: object):
    """Serialize datetime objects for JSON."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def main():
    run_id = os.environ.get("KERNELBOT_RUN_ID", "unknown")
    payload_b64 = os.environ.get("KERNELBOT_PAYLOAD")

    print("=== Kernelbot Evaluation ===")
    print(f"Run ID: {run_id}")
    print(f"GPU: {os.environ.get('NVIDIA_VISIBLE_DEVICES', 'not set')}")
    print(f"GPU Index: {os.environ.get('KERNELBOT_GPU_INDEX', 'not set')}")
    print()

    if not payload_b64:
        # No payload means this was triggered by push/PR, not API
        # Exit gracefully so CI doesn't fail
        print("KERNELBOT_PAYLOAD not set - this build was triggered by push/PR, not API.")
        print("Skipping evaluation. To run an evaluation, trigger via BuildkiteLauncher API.")
        print()
        print("=== Skipped (no payload) ===")
        sys.exit(0)

    # Decode payload
    try:
        compressed = base64.b64decode(payload_b64)
        config_json = zlib.decompress(compressed).decode("utf-8")
        config = json.loads(config_json)
    except Exception as e:
        print(f"ERROR: Failed to decode payload: {e}", file=sys.stderr)
        sys.exit(1)

    # Import here to catch import errors clearly
    from libkernelbot.run_eval import run_config

    # Run evaluation
    print("Starting evaluation...")
    result = run_config(config)

    # Write result
    result_dict = asdict(result)
    result_json = json.dumps(result_dict, default=serialize, indent=2)
    Path("result.json").write_text(result_json)
    print("Result written to result.json")

    # Print summary
    print()
    print("=== Result ===")
    print(f"Success: {result.success}")
    if result.error:
        print(f"Error: {result.error}")

    sys.exit(0 if result.success else 1)


if __name__ == "__main__":
    main()
