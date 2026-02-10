import signal
import traceback
from contextlib import contextmanager

from modal import App, Image, Volume

from libkernelbot.run_eval import FullResult, SystemInfo, run_config

# Create a stub for the Modal app
# IMPORTANT: This has to stay in separate file or modal breaks
app = App("discord-bot-runner")
cuda_version = "13.1.0"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

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
#   C++ headers installed on the image (like CUTLASS) require explicit include paths:
#     load_inline(
#         ...
#         extra_include_paths=["/opt/cutlass/include", "/opt/cutlass/tools/util/include"],
#     )
#   For raw nvcc compilation, CPLUS_INCLUDE_PATH is set so includes work automatically.
#
cuda_image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.13")
    .run_commands("ln -sf $(which python) /usr/local/bin/python3")
    .apt_install(
        "git",
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
    .uv_pip_install(
        "torch==2.9.1",
        "torchvision",
        "torchaudio",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    # other frameworks
    .uv_pip_install(
        "tinygrad~=0.10",
        "helion",
    )
    # nvidia cuda packages
    .uv_pip_install(
        "nvidia-cupynumeric~=25.3",
        "nvidia-cutlass-dsl==4.3.5",
        "cuda-core[cu13]",
        "cuda-python[all]==13.0",
        # "nvmath-python[cu13]~=0.4",
        # "numba-cuda[cu13]~=0.15",
    )
    # CUTLASS C++ headers for #include <cutlass/...>
    .run_commands(
        "git clone --depth 1 --branch v4.3.5 https://github.com/NVIDIA/cutlass.git /opt/cutlass",
    )
    .env({
        "CUTLASS_PATH": "/opt/cutlass",
        "CPLUS_INCLUDE_PATH": "/opt/cutlass/include:/opt/cutlass/tools/util/include",
    })
)

cuda_image = cuda_image.add_local_python_source(
    "libkernelbot",
    "modal_runner",
    "modal_runner_archs",
)

# === Model Competition Image ===
#
# For e2e model competitions where users submit vLLM forks.
# Includes all vLLM dependencies but NOT vllm itself (the user's fork replaces it).
# sccache caches CUDA extension compilation across submissions.
#
model_image = (
    Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.13")
    .run_commands("ln -sf $(which python) /usr/local/bin/python3")
    .apt_install("git", "gcc-13", "g++-13")
    .pip_install(
        "torch==2.9.1",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        "numpy",
        "transformers",
        "tokenizers",
        "huggingface_hub",
        "ray",
        "uvicorn",
        "fastapi",
        "pydantic",
        "aiohttp",
        "requests",
        "packaging",
        "ninja",
        "wheel",
        "sccache",
    )
    # Install vLLM to pull in all transitive deps, then uninstall vllm itself.
    # The user's fork will be pip installed at runtime.
    .run_commands(
        "pip install vllm && pip uninstall vllm -y",
    )
    .env({
        "SCCACHE_DIR": "/sccache",
        "CMAKE_C_COMPILER_LAUNCHER": "sccache",
        "CMAKE_CXX_COMPILER_LAUNCHER": "sccache",
    })
)

model_image = model_image.add_local_python_source(
    "libkernelbot",
    "modal_runner",
    "modal_runner_archs",
)

# === Volumes ===
model_weights = Volume.from_name("model-weights", create_if_missing=True)
sccache_vol = Volume.from_name("sccache", create_if_missing=True)


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
    timeout_seconds: int = 600,
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
