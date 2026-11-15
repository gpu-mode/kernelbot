"""Shared, stdlib-only identity rules for trusted NVCC header profiles."""

import hashlib
import json
import os
import re
from pathlib import Path

VERSION = 1
COMPILER = "/usr/local/cuda/bin/nvcc"
FINGERPRINT = Path("/opt/kernelbot-pch/nvcc-fingerprint")
# Match PyTorch 2.12 load_inline's CUDA prefix exactly. In particular, ATen is
# included transitively by torch/types.h rather than inserted before it here.
HEADERS = ("torch/types.h", "cuda.h", "cuda_runtime.h")
HEADER = "#pragma once\n" + "".join(f"#include <{h}>\n" for h in HEADERS)
HEADER += "#pragma hdrstop\n"
HEADER_SHA256 = hashlib.sha256(HEADER.encode()).hexdigest()
ARCHES = ("75", "80", "89", "90a", "100")
# Ordered tuples observed in the 2026-11-13 submission snapshot. Preserve every
# option and its position; the warmer captures PyTorch's remaining flags.
FLAG_PROFILES = {
    "default": [],
    "O3": ["-O3"],
    "O3-fast-math": ["-O3", "--use_fast_math"],
}
PROFILES = {
    f"sm{arch}-{name}": (arch, flags) for arch in ARCHES for name, flags in FLAG_PROFILES.items()
}
EXPLICIT_PROFILES = {
    "sm100a-O3-fast-math-arch": ("100a", ["-O3", "--use_fast_math", "-arch=sm_100a"]),
    "sm100-O3-arch-fast-math": ("100", ["-O3", "-arch=sm_100", "--use_fast_math"]),
    "sm80-O3-cxx17-arch-fast-math": (
        "80",
        ["-O3", "-std=c++17", "-arch=sm_80", "--use_fast_math"],
    ),
}
PROFILES.update(EXPLICIT_PROFILES)
DEFAULT_PROFILES = tuple(PROFILES)
ENVIRONMENT_KEYS = (
    "CPATH",
    "CPLUS_INCLUDE_PATH",
    "C_INCLUDE_PATH",
    "GCC_EXEC_PREFIX",
    "COMPILER_PATH",
    "NVCC_CCBIN",
    "NVVMIR_LIBRARY_DIR",
    "PATH",
    "LD_LIBRARY_PATH",
    "LIBRARY_PATH",
    "CUDA_HOME",
    "CUDA_PATH",
    "CUDA_ROOT",
)
HEX = re.compile(r"[0-9a-f]{64}\Z")
PCH_NAME = re.compile(r"(?:cicc|cudafe)_[0-9a-f_]+\.pch\Z")
PREFIX = re.compile(
    r"\A[ \t\r\n]*"
    + "".join(r"#[ \t]*include[ \t]*<" + re.escape(h) + r">[ \t]*\r?\n[ \t\r\n]*" for h in HEADERS)
)


def cache_root():
    return Path(os.environ.get("KERNELBOT_NVCC_PCH_ROOT", "/kernelbot-nvcc-pch"))


def generation():
    value = FINGERPRINT.read_text().strip()
    if not HEX.fullmatch(value):
        raise ValueError("Invalid NVCC toolchain fingerprint")
    return value


def environment():
    return {key: os.environ.get(key) for key in ENVIRONMENT_KEYS}


def compile_flags(args):
    """Return only compatible single CUDA compile flags; never mutate the command."""
    if args.count("-c") != 1 or args.count("-o") != 1:
        return None
    if any(os.environ.get(k) for k in ("NVCC_PREPEND_FLAGS", "NVCC_APPEND_FLAGS")):
        return None
    if any(
        a.startswith(
            (
                "@",
                "-ccbin",
                "--compiler-bindir",
                "--pch",
                "--use-pch",
                "--create-pch",
                "-include",
                "--pre-include",
            )
        )
        for a in args
    ):
        return None
    try:
        source = Path(args[args.index("-c") + 1])
        if source.suffix != ".cu":
            return None
        # PyTorch inserts this prefix before user code. Do not move user macros or
        # introduce Torch/CUTLASS headers into deliberately minimal CUDA sources.
        with source.open() as stream:
            if not PREFIX.match(stream.read(4096)):
                return None
    except (IndexError, OSError, UnicodeError):
        return None
    return strip_outputs(args)


def strip_outputs(args, drop_module=True):
    flags = []
    paired = {"-c", "-o", "-MF", "-MT", "-MQ", "--dependency-output"}
    standalone = {"-MD", "-MMD", "--generate-dependencies-with-compile"}
    index = 0
    while index < len(args):
        arg = args[index]
        if arg in paired:
            if index + 1 >= len(args):
                return None
            index += 2
            continue
        if drop_module and re.fullmatch(r"-DTORCH_EXTENSION_NAME=[A-Za-z_]\w*", arg):
            # Keep this macro in the real invocation. NVIDIA validates whether
            # it was used in a PCH; the curated forwarding header does not use it.
            index += 1
            continue
        if arg not in standalone and not arg.startswith("--dependency-output="):
            flags.append(arg)
        index += 1
    return flags


def identity(flags, fingerprint):
    return {
        "version": VERSION,
        "generation": fingerprint,
        "header_sha256": HEADER_SHA256,
        "flags": flags,
        "environment": environment(),
    }


def cache_key(flags, fingerprint):
    value = json.dumps(identity(flags, fingerprint), sort_keys=True).encode()
    return hashlib.sha256(value).hexdigest()


def read_manifest(entry, flags, fingerprint):
    """Validate ready metadata and cheap file stats, without reading multi-GB blobs."""
    manifest = json.loads((entry / "manifest.json").read_text())
    return validate_manifest(manifest, entry, flags, fingerprint)


def validate_manifest(manifest, entry, flags, fingerprint):
    if not isinstance(manifest, dict):
        raise ValueError("Invalid PCH manifest")
    expected = identity(flags, fingerprint)
    if any(manifest.get(k) != value for k, value in expected.items()):
        raise ValueError("PCH profile identity changed")
    if manifest.get("key") != entry.name or entry.name != cache_key(flags, fingerprint):
        raise ValueError("PCH profile key changed")
    if manifest.get("profile") not in PROFILES:
        raise ValueError("Unrecognized trusted PCH profile")
    if (entry / "gpumode.cuh").read_text() != HEADER:
        raise ValueError("PCH forwarding header changed")
    validate_files(entry, manifest.get("files", []))
    return manifest


def validate_files(entry, files):
    if len(files) != 2:
        raise ValueError("PCH profile must contain both frontends")
    frontends = set()
    for item in files:
        path = Path(item["path"])
        if path.parent != Path("pch") or not PCH_NAME.fullmatch(path.name):
            raise ValueError("Invalid PCH manifest path")
        target = entry / path
        if not target.resolve().is_relative_to(entry.resolve()):
            raise ValueError("PCH file escapes its profile")
        size = item["bytes"]
        if not isinstance(size, int) or size <= 0 or target.stat().st_size != size:
            raise ValueError("PCH file is missing or truncated")
        frontends.add(path.name.split("_", 1)[0])
    if frontends != {"cicc", "cudafe"}:
        raise ValueError("PCH profile is missing a frontend")
