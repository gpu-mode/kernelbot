#!/usr/bin/env python3
"""Read-only, opt-in NVCC PCH wrapper for compatible load_inline CUDA sources."""

import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import nvcc_common as common

USED = re.compile(r'using precompiled header file "([^"]+)"')
CREATED = re.compile(r'creating precompiled header file "([^"]+)"')
REJECTION = re.compile(
    r"invalid (?:PCH|precompiled header)|cannot open precompiled header|"
    r"precompiled header[^\n]*(?:invalid|incomplete|not generated|have changed)|"
    r"memory usage conflict with precompiled header|"
    r"command line options do not match those used when precompiled header|"
    r"initial sequence of preprocessing directives is not compatible|"
    r"could not publish precompiled header|"
    r"source preceding the #pragma hdrstop does not match|"
    r"incompatible precompiled header \(PCH\)",
    re.IGNORECASE,
)


def event(status, started, code, **details):
    record = {
        "status": status,
        "seconds": time.perf_counter() - started,
        "returncode": code,
        **details,
    }
    print("[kernelbot-nvcc-pch] " + json.dumps(record, sort_keys=True), file=sys.stderr)
    if destination := os.environ.get("KERNELBOT_NVCC_PCH_EVENTS"):
        try:
            with open(destination, "a") as output:
                output.write(json.dumps(record, sort_keys=True) + "\n")
        except OSError:
            pass  # Observability must not break a submission compilation.


def run_compiler(args, capture=False):
    return subprocess.run([common.COMPILER, *args], capture_output=capture, text=capture)


def passthrough(args, status, reason, started, **details):
    result = run_compiler(args)
    if "-c" in args:
        event(status, started, result.returncode, reason=reason, **details)
    return result.returncode


def consumed(log, entry):
    paths = USED.findall(log)
    frontends = {Path(path).name.split("_", 1)[0] for path in paths}
    valid = (
        frontends == {"cicc", "cudafe"}
        and not CREATED.search(log)
        and all(Path(path).is_relative_to(entry / "pch") for path in paths)
    )
    return valid, sorted(frontends)


def main(args):
    started = time.perf_counter()
    if (
        os.environ.get("KERNELBOT_NVCC_PCH") != "1"
        or os.environ.get("KERNELBOT_NVCC_PCH_DISABLE") == "1"
    ):
        return passthrough(args, "disabled", "disabled", started)
    flags = common.compile_flags(args)
    if flags is None:
        return passthrough(args, "bypass", "unsupported_invocation_or_headers", started)
    details = {}
    try:
        fingerprint = common.generation()
        key = common.cache_key(flags, fingerprint)
        details = {"generation": fingerprint, "key": key}
        entry = common.cache_root() / fingerprint / key
        manifest = common.read_manifest(entry, flags, fingerprint)
    except (OSError, ValueError, KeyError, TypeError):
        return passthrough(args, "miss", "profile_unavailable", started, **details)
    details["profile"] = manifest["profile"]
    cached_args = [
        *args,
        "-include",
        str(entry / "gpumode.cuh"),
        "--pch",
        "--pch-dir",
        str(entry / "pch"),
        "--pch-verbose",
    ]
    tick = time.perf_counter()
    result = run_compiler(cached_args, capture=True)
    cached_seconds = time.perf_counter() - tick
    log = result.stdout + result.stderr
    hit, frontends = consumed(log, entry)
    if result.returncode and (CREATED.search(log) or REJECTION.search(log)):
        # Cache-specific failure only. Genuine user compiler errors remain errors.
        # Original argv removes both the forwarding header and every PCH option.
        print("[kernelbot-nvcc-pch] rejected profile; compiling normally", file=sys.stderr)
        return passthrough(
            args,
            "fallback",
            "cache_rejected",
            started,
            cached_attempt_seconds=cached_seconds,
            **details,
        )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    status = "error" if result.returncode else ("hit" if hit else "fallback")
    event(
        status,
        started,
        result.returncode,
        frontends=frontends,
        reason="consumed" if hit else "not_consumed",
        **details,
    )
    return result.returncode


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
