#!/usr/bin/env python3
"""Transparent GCC wrapper for the trusted, read-only torch header PCH Volume.

Only the first implicit load_inline header is cached. Kernel objects and shared
libraries stay in the submission container's normal torch extension cache.
"""

import hashlib
import json
import os
import re
import subprocess
import sys
import time
import uuid
from pathlib import Path

COMPILER = "/usr/bin/g++"
ROOT = Path(os.environ.get("KERNELBOT_PCH_ROOT", "/kernelbot-pch"))
FINGERPRINT = Path("/opt/kernelbot-pch/fingerprint")


def compile_flags(args):
    """Return preprocessing/compilation flags, excluding per-module build outputs."""
    if args.count("-c") != 1 or "-o" not in args:
        return None
    if args.index("-c") + 1 == len(args):
        return None
    source = args[args.index("-c") + 1]
    if not source.endswith(".cpp"):
        return None
    # Do not change header order or inject torch into no_implicit_headers users.
    try:
        prefix = Path(source).read_text()
    except (OSError, UnicodeError):
        return None
    if not re.match(r"\s*#\s*include\s*<torch/extension\.h>", prefix):
        return None
    flags = []
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in {"-c", "-o", "-MF", "-MT", "-MQ"}:
            index += 2
            continue
        if arg not in {"-MMD", "-MD", "-H"} and not arg.startswith("-DTORCH_EXTENSION_NAME="):
            flags.append(arg)
        index += 1
    return flags


def cache_key(flags, fingerprint):
    # Include header-search environment as well as argv. Never share incompatible
    # macro, optimization, ABI, language-standard or include-path configurations.
    payload = [
        fingerprint,
        flags,
        {
            k: os.environ.get(k)
            for k in (
                "CPATH",
                "CPLUS_INCLUDE_PATH",
                "C_INCLUDE_PATH",
                "GCC_EXEC_PREFIX",
                "COMPILER_PATH",
            )
        },
    ]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def make_fingerprint():
    """Run once when building the immutable image, never on a submission's GPU."""
    import sysconfig

    import torch

    digest = hashlib.sha256()
    digest.update(Path(__file__).read_bytes())
    digest.update(subprocess.check_output([COMPILER, "-v"], stderr=subprocess.STDOUT))
    digest.update(sys.version.encode())
    digest.update(torch.__version__.encode())
    digest.update(Path("/etc/os-release").read_bytes())
    # Hash installed headers, including CUDA and compiler headers, so rebuilding
    # the image with changed header contents cannot reuse stale PCHs.
    roots = [
        Path(torch.__file__).parent / "include",
        Path(sysconfig.get_path("include")),
        Path("/usr/include"),
        *Path("/usr/lib/gcc").glob("*/*/include*"),
        Path("/usr/local/cuda/include"),
        Path("/opt/cutlass/include"),
        Path("/opt/cutlass/tools/util/include"),
        Path("/opt/mathdx/include"),
        Path("/opt/mathdx/external/cutlass/include"),
    ]
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                digest.update(str(path).encode())
                digest.update(path.read_bytes())
    FINGERPRINT.write_text(digest.hexdigest())


def run_compiler(args):
    trace = os.environ.get("KERNELBOT_PCH_TRACE") == "1" and "-c" in args
    started = time.perf_counter()
    result = subprocess.call([COMPILER, *args, *(["-H"] if trace and "-include" in args else [])])
    if trace:
        print(f"[kernelbot-pch] compile_seconds={time.perf_counter() - started:.3f}", flush=True)
    return result


def main(args):
    if args == ["--kernelbot-fingerprint"]:
        make_fingerprint()
        return 0
    if os.environ.get("KERNELBOT_PCH_DISABLE") == "1":
        return run_compiler(args)
    flags = compile_flags(args)
    if flags is None:
        return run_compiler(args)
    key = cache_key(flags, FINGERPRINT.read_text())
    entry = ROOT / key
    header = entry / "torch.h"
    pch = entry / "torch.h.gch"
    # Write mode exists only on the trusted CPU warmer, never the GPU runner.
    if not pch.exists() and os.environ.get("KERNELBOT_PCH_WRITE") == "1":
        entry.mkdir(parents=True, exist_ok=True)
        header.write_text("#include <torch/extension.h>\n")
        temporary = entry / f"torch.h.gch.{uuid.uuid4().hex}.tmp"
        subprocess.run(
            [COMPILER, *flags, "-x", "c++-header", str(header), "-o", str(temporary)], check=True
        )
        temporary.replace(pch)
        print(f"[kernelbot-pch] built {key}", flush=True)
    if pch.is_file():
        print(f"[kernelbot-pch] hit {key}", flush=True)
        return run_compiler([*args, "-include", str(header), "-Winvalid-pch"])
    print(f"[kernelbot-pch] miss {key}; compiling normally", flush=True)
    return run_compiler(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
