"""
Buildkite runner script for kernel submissions.

This script runs inside a Docker container on Buildkite agents.
It reads the submission payload from the SUBMISSION_PAYLOAD environment variable,
executes the kernel, and writes results to result.json for artifact upload.

The agent is pre-configured with:
- CUDA_VISIBLE_DEVICES bound to a single GPU
- CPU/RAM limits via systemd cgroups
"""

import base64
import json
import os
import zlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from libkernelbot.run_eval import run_config


def serialize(obj: object):
    """JSON serializer for objects not serializable by default."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def main():
    # Get payload from environment variable
    payload_b64 = os.environ.get("SUBMISSION_PAYLOAD")
    if not payload_b64:
        raise RuntimeError("SUBMISSION_PAYLOAD environment variable not set")

    # Decompress and parse config
    payload = zlib.decompress(base64.b64decode(payload_b64)).decode("utf-8")
    config = json.loads(payload)

    # Run the submission
    result = run_config(config)

    # Write result to file for artifact upload
    result_dict = asdict(result)
    Path("result.json").write_text(json.dumps(result_dict, default=serialize))

    # Create profile_data directory if profiling was enabled
    # (profile artifacts will be written there by run_config)
    profile_dir = Path("profile_data")
    if profile_dir.exists():
        print(f"Profile data available in {profile_dir}")


if __name__ == "__main__":
    main()
