#!/usr/bin/env python3
"""
Northflank runner script for executing kernel benchmarks.

This script is executed inside a Northflank job container. It:
1. Reads the payload from environment variables
2. Decompresses and parses the submission config
3. Runs the benchmark
4. Outputs results both as a file and to stdout (for log parsing)

Environment variables expected:
- PAYLOAD: Base64-encoded, zlib-compressed JSON config
- REQUIREMENTS: Python packages to install (handled by job setup)
- RUN_ID: Unique identifier for this run
- RUNNER: (AMD only) Runner type for GPU selection
"""

import base64
import json
import os
import sys
import zlib
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from libkernelbot.run_eval import run_config


def serialize(obj: object):
    """Custom JSON serializer for datetime objects."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


def main():
    """Execute the benchmark and output results."""

    # Get payload from environment variable
    payload_b64 = os.environ.get("PAYLOAD")
    if not payload_b64:
        print("ERROR: PAYLOAD environment variable not set", file=sys.stderr)
        sys.exit(1)

    run_id = os.environ.get("RUN_ID", "unknown")
    print(f"Starting Northflank run {run_id}")

    # Decompress and parse the payload
    try:
        payload = zlib.decompress(base64.b64decode(payload_b64)).decode("utf-8")
        config = json.loads(payload)
        print(f"Loaded config: {config.get('problem', 'unknown')} in {config.get('mode', 'unknown')} mode")
    except Exception as e:
        print(f"ERROR: Failed to parse payload: {e}", file=sys.stderr)
        sys.exit(1)

    # Run the benchmark
    try:
        print("Running benchmark...")
        result = asdict(run_config(config))
        print("Benchmark completed successfully")
    except Exception as e:
        print(f"ERROR: Benchmark failed: {e}", file=sys.stderr)
        # Create a minimal error result
        result = {
            "success": False,
            "error": str(e),
            "runs": {},
            "system": {}
        }

    # Serialize the result
    result_json = json.dumps(result, default=serialize, indent=2)

    # Write to local file
    result_path = Path("result.json")
    result_path.write_text(result_json)
    print(f"Wrote result to {result_path}")

    # Upload to presigned URL
    upload_url = os.getenv("RESULT_UPLOAD_URL")
    if upload_url:
        try:
            import requests

            # Upload via HTTP PUT to presigned URL
            with open(result_path, 'rb') as f:
                response = requests.put(
                    upload_url,
                    data=f,
                    headers={'Content-Type': 'application/json'}
                )

            if response.status_code in (200, 201, 204):
                print(f"Successfully uploaded result to storage")
            else:
                print(f"WARNING: Upload failed with status {response.status_code}: {response.text}", file=sys.stderr)

        except Exception as e:
            print(f"ERROR: Failed to upload result: {e}", file=sys.stderr)
            # Don't fail the job if upload fails
    else:
        print("WARNING: RESULT_UPLOAD_URL not set, skipping upload", file=sys.stderr)

    # Exit with appropriate code
    if result.get("success", False):
        print(f"Run {run_id} completed successfully")
        sys.exit(0)
    else:
        print(f"Run {run_id} completed with errors", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
