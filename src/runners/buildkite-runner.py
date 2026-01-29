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
import sys
import traceback
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


def write_error_result(error_message: str):
    """Write an error result to result.json when execution fails."""
    error_result = {
        "success": False,
        "error": error_message,
        "runs": {},
        "system": {},
    }
    Path("result.json").write_text(json.dumps(error_result, default=serialize))


def main():
    try:
        # Get payload from environment variable
        payload_b64 = os.environ.get("SUBMISSION_PAYLOAD")
        if not payload_b64:
            write_error_result("SUBMISSION_PAYLOAD environment variable not set")
            sys.exit(1)

        # Decompress and parse config
        try:
            payload = zlib.decompress(base64.b64decode(payload_b64)).decode("utf-8")
            config = json.loads(payload)
        except Exception as e:
            write_error_result(f"Failed to decompress/parse payload: {e}")
            sys.exit(1)

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

    except Exception as e:
        # Catch any unexpected errors and write them to result.json
        error_msg = f"Runner error: {e}\n{traceback.format_exc()}"
        print(error_msg, file=sys.stderr)
        write_error_result(error_msg)
        sys.exit(1)


if __name__ == "__main__":
    main()
