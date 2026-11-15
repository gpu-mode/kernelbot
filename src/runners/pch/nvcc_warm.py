"""Trusted CPU-only NVCC PCH warmer; accepts curated profile names, never user flags."""

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import nvcc_common as common

CAPTURE = """#!/usr/bin/env python3
import json, os, subprocess, sys
from pathlib import Path
compiler = "/usr/local/cuda/bin/nvcc" if sys.argv[0].endswith("nvcc") else "/usr/bin/g++"
if "-c" in sys.argv:
    if compiler.endswith("nvcc"):
        with open(os.environ["KERNELBOT_CAPTURE_NVCC"], "w") as output:
            json.dump(sys.argv[1:], output)
        sys.exit(1)
    Path(sys.argv[sys.argv.index("-o") + 1]).touch()
    sys.exit(0)
sys.exit(subprocess.call([compiler, *sys.argv[1:]]))
"""
BUILD = """
import json, os
from torch.utils.cpp_extension import load_inline
load_inline(name="kernelbot_nvcc_warmer", cpp_sources="int warm() { return 1; }",
    cuda_sources="__global__ void warm_cuda() {}", functions=["warm"],
    extra_cuda_cflags=json.loads(os.environ["KERNELBOT_CAPTURE_FLAGS"]),
    build_directory=os.environ["KERNELBOT_CAPTURE_BUILD"], verbose=True)
"""


def torch_arch(arch):
    match = re.fullmatch(r"([1-9][0-9]*)([0-9])(a?)", arch)
    if match is None:
        raise ValueError("Invalid curated CUDA architecture")
    major, minor, suffix = match.groups()
    return f"{major}.{minor}{suffix}"


def capture(profile, work):
    arch, extra = common.PROFILES[profile]
    for kind in ("cxx", "nvcc"):
        path = work / f"capture-{kind}"
        path.write_text(CAPTURE)
        path.chmod(0o755)
    capture_path = work / "args.json"
    environment = os.environ | {
        "CXX": str(work / "capture-cxx"),
        "PYTORCH_NVCC": str(work / "capture-nvcc"),
        "TORCH_CUDA_ARCH_LIST": torch_arch(arch),
        "MAX_JOBS": "2",
        "KERNELBOT_CAPTURE_NVCC": str(capture_path),
        "KERNELBOT_CAPTURE_FLAGS": json.dumps(extra),
        "KERNELBOT_CAPTURE_BUILD": str(work),
    }
    result = subprocess.run(
        ["python3", "-c", BUILD], env=environment, capture_output=True, text=True, timeout=180
    )
    if not capture_path.exists():
        raise RuntimeError("Could not capture PyTorch NVCC flags: " + result.stdout + result.stderr)
    args = json.loads(capture_path.read_text())
    flags = common.compile_flags(args)
    if flags is None:
        raise RuntimeError("Trusted warm invocation did not pass runtime eligibility rules")
    return flags, common.strip_outputs(args, drop_module=False)


def warm_profile(profile, fingerprint):
    with tempfile.TemporaryDirectory(prefix="kernelbot-nvcc-warm-") as directory:
        work = Path(directory)
        flags, warm_flags = capture(profile, work)
        key = common.cache_key(flags, fingerprint)
        entry = common.cache_root() / fingerprint / key
        if (entry / "manifest.json").exists():
            return common.read_manifest(entry, flags, fingerprint)
        # Only incomplete, unpublished entries may be repaired by the single writer.
        entry.mkdir(parents=True, exist_ok=True)
        shutil.rmtree(entry / "pch", ignore_errors=True)
        (entry / "pch").mkdir()
        (entry / "gpumode.cuh").write_text(common.HEADER)
        command = [
            common.COMPILER,
            *warm_flags,
            "--pch",
            "--pch-verbose",
            "--pch-dir",
            str(entry / "pch"),
            "-x",
            "cu",
            "-c",
            str(entry / "gpumode.cuh"),
            "-o",
            str(work / "header.o"),
        ]
        print(f"Warming {profile}: {key}", flush=True)
        started = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise RuntimeError(result.stdout + result.stderr)
        files = [
            {"path": str(path.relative_to(entry)), "bytes": path.stat().st_size}
            for path in sorted((entry / "pch").glob("*.pch"))
        ]
        manifest = common.identity(flags, fingerprint) | {
            "key": key,
            "profile": profile,
            "files": files,
            "creation_seconds": time.perf_counter() - started,
        }
        common.validate_manifest(manifest, entry, flags, fingerprint)
        temporary = entry / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        # Validate the exact ready metadata before making it visible to workers.
        # The parent Modal warmer commits the completed volume snapshot afterward.
        temporary.replace(entry / "manifest.json")
        print(json.dumps(manifest), flush=True)
        return manifest


def warm(profiles=()):
    selected = list(profiles) or list(common.DEFAULT_PROFILES)
    if not set(selected) <= common.PROFILES.keys():
        raise ValueError(f"Choose fixed profiles from {sorted(common.PROFILES)}")
    if any(os.environ.get(k) for k in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")):
        raise ValueError("Trusted warmup does not accept NVCC environment flags")
    fingerprint = common.generation()
    return [warm_profile(profile, fingerprint) for profile in selected]


if __name__ == "__main__":
    warm(sys.argv[1:])
