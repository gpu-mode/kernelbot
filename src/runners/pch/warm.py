"""Trusted CPU-only header warmup. No submission source or flags are accepted."""

import os
import sys
import tempfile

# Keep the common profiles bounded. Unsupported flags simply compile normally.
PROFILES = {"default": [], "O2": ["-O2"], "O3": ["-O3"], "O3-fast-math": ["-O3", "-ffast-math"]}


def warm(profiles=()):
    from torch.utils.cpp_extension import load_inline

    os.environ["KERNELBOT_PCH_WRITE"] = "1"
    os.environ["MAX_JOBS"] = "1"
    # PyTorch computes CUDA flags even for a C++-only warmup with with_cuda=True.
    # The host PCH is architecture-independent; no GPU or CUDA source is needed.
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    for with_cuda in (False, True):
        for index, (name, flags) in enumerate(PROFILES.items()):
            if profiles and f"{'cuda' if with_cuda else 'cpu'}-{name}" not in profiles:
                continue
            with tempfile.TemporaryDirectory() as build:
                load_inline(
                    name=f"kernelbot_pch_warm_{int(with_cuda)}_{index}",
                    cpp_sources="int warm() { return 1; }",
                    functions=["warm"],
                    extra_cflags=flags,
                    with_cuda=with_cuda,
                    build_directory=build,
                    verbose=True,
                )


if __name__ == "__main__":
    warm(sys.argv[1:])
