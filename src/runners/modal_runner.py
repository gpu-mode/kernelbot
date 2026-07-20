import signal
import traceback
from contextlib import contextmanager

from modal import App, Image

from libkernelbot.run_eval import FullResult, SystemInfo, run_config

# Create a stub for the Modal app
# IMPORTANT: This has to stay in separate file or modal breaks
app = App("discord-bot-runner")
cuda_version = "13.3.0"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

mathdx_version = "26.06.0"
mathdx_archive = f"nvidia-mathdx-{mathdx_version}-cuda13.tar.gz"
mathdx_url = (
    "https://developer.download.nvidia.com/compute/cublasdx/redist/"
    f"cublasdx/cuda13/{mathdx_archive}"
)
mathdx_sha256 = "042b7c57a636c271cca32dffcc0a822ed6b2abc0b8ef5703ab2445d58563a1e6"

# === Image Definition ===
#
# Adding new C++ library dependencies:
#   1. Add a .run_commands() step that installs headers to /opt/<library_name>
#      Use `git clone --depth 1 --branch <tag>` for header-only libs to keep the image small.
#   2. Add the include paths to CPLUS_INCLUDE_PATH in the .env() block at the bottom
#      so that nvcc finds them automatically without -I flags.
#   3. Test changes with test_cutlass_image.py (or a similar script) before deploying:
#        cd src/runners && modal run test_cutlass_image.py
#
# For users writing submissions with torch.utils.cpp_extension.load_inline:
#   MathDx is already on CPLUS_INCLUDE_PATH. Include cuBLASDx in cuda_sources,
#   which load_inline compiles with nvcc, rather than in cpp_sources:
#     load_inline(
#         ...
#         cuda_sources=cuda_source,  # #include <cublasdx.hpp> goes here
#     )
#   For raw nvcc compilation, CPLUS_INCLUDE_PATH is set so includes work automatically.
#
cuda_image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.13")
    .run_commands("ln -sf $(which python) /usr/local/bin/python3")
    .apt_install(
        "git",
        "curl",
        "gcc-13",
        "g++-13",
        "clang-18",
    )
    .uv_pip_install(
        "ninja~=1.11",
        "wheel~=0.45",
        "requests~=2.32.4",
        "packaging~=25.0",
        "numpy~=2.3",
        "pytest",
        "PyYAML",
    )
    # other frameworks
    .uv_pip_install(
        "tinygrad~=0.10",
        "helion",
    )
    # nvidia cuda packages
    .uv_pip_install(
        "nvidia-cutlass-dsl==4.5.2",
        "cuda-core[cu13]",
        "cuda-python[all]==13.0",
        # cuTile: the CUDA Tile programming model in Python (`import cuda.tile`).
        # CUDA 13.3 supplies tileiras; the extra's toolkit constraint conflicts
        # with the CUDA 13.0 dependency set used by the PyTorch wheel.
        "cuda-tile==1.4.0",
        "nvmath-python[cu13-dx]==0.9.0",
        "nvidia-libmathdx-cu13==0.3.2.6",
        "cuda-toolkit[cccl,nvrtc]==13.0.2",
        # "numba-cuda[cu13]~=0.15",
    )
    # Install torch last so its CUDA/NCCL dependency set wins over broader CUDA Python packages.
    .uv_pip_install(
        "torch==2.12.0",
    )
    # CUTLASS C++ headers for #include <cutlass/...>
    .run_commands(
        "git clone --depth 1 --branch v4.5.2 https://github.com/NVIDIA/cutlass.git /opt/cutlass",
        (
            f"curl -fsSL {mathdx_url} -o /tmp/{mathdx_archive} && "
            f"echo '{mathdx_sha256}  /tmp/{mathdx_archive}' | sha256sum -c - && "
            "mkdir -p /opt/mathdx && "
            f"tar -xzf /tmp/{mathdx_archive} --strip-components=4 -C /opt/mathdx && "
            f"rm /tmp/{mathdx_archive}"
        ),
    )
    .env({
        "CUTLASS_PATH": "/opt/cutlass",
        "MATHDX_HOME": "/opt/mathdx",
        "CPLUS_INCLUDE_PATH": (
            "/opt/mathdx/include:/opt/mathdx/external/cutlass/include:"
            "/opt/cutlass/include:/opt/cutlass/tools/util/include"
        ),
    })
    .run_commands(
        "python -m pip check",
        'python -c "import cuda.tile, nvmath"',
        "tileiras --version",
        (
            "printf '#include <cublasdx.hpp>\\n' | "
            "nvcc -std=c++17 -x cu -c - -o /tmp/cublasdx-smoke.o && "
            "rm /tmp/cublasdx-smoke.o"
        ),
    )
)

cuda_image = cuda_image.add_local_python_source(
    "libkernelbot",
    "modal_runner",
    "modal_runner_archs",
)

MODAL_RUN_TIMEOUT_SECONDS = 60 * 60


class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds: int):
    """Context manager that raises TimeoutException after specified seconds"""

    def timeout_handler(signum, frame):
        raise TimeoutException(f"Script execution timed out after {seconds} seconds")

    # Set up the signal handler
    original_handler = signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(seconds)

    try:
        yield
    finally:
        # Restore the original handler and disable the alarm
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)


def modal_run_config(  # noqa: C901
    config: dict,
    timeout_seconds: int = MODAL_RUN_TIMEOUT_SECONDS,
) -> FullResult:
    """Modal version of run_pytorch_script, handling timeouts"""
    try:
        with timeout(timeout_seconds):
            return run_config(config)
    except TimeoutException as e:
        return FullResult(
            success=False,
            error=f"Timeout Error: {str(e)}",
            runs={},
            system=SystemInfo(),
        )
    except Exception as e:
        exception = "".join(traceback.format_exception(e))
        return FullResult(
            success=False,
            error=f"Error executing script:\n{exception}",
            runs={},
            system=SystemInfo(),
        )
