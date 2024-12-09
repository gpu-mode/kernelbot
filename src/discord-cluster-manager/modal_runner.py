import modal
from modal import App, Image
from contextlib import contextmanager
import signal
from utils import strip_imports

# Create a stub for the Modal app
# IMPORTANT: This has to stay in separate file or modal breaks
modal_app = App("discord-bot-runner")


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


@modal_app.function(
    gpu="T4",
    image=Image.debian_slim(python_version="3.10").pip_install(["torch", "numpy"]),
)
def run_pytorch_script_t4(
    script_content: str,
    gpu_type: str,
    eval_content: str = None,
    reference_content: str = None,
    timeout_seconds: int = 300,
) -> tuple[str, float]:
    return run_pytorch_script(
        script_content,
        gpu_type,
        eval_content,
        reference_content,
        timeout_seconds,
    )


@modal_app.function(
    gpu="L4",
    image=Image.debian_slim(python_version="3.10").pip_install(["torch", "numpy"]),
)
def run_pytorch_script_l4(
    script_content: str,
    gpu_type: str,
    eval_content: str = None,
    reference_content: str = None,
    timeout_seconds: int = 300,
) -> tuple[str, float]:
    return run_pytorch_script(
        script_content,
        gpu_type,
        eval_content,
        reference_content,
        timeout_seconds,
    )


@modal_app.function(
    gpu=modal.gpu.A100(size="80GB"),
    image=Image.debian_slim(python_version="3.10").pip_install(["torch"]),
)
def run_pytorch_script_a100_80gb(
    script_content: str,
    gpu_type: str,
    eval_content: str = None,
    reference_content: str = None,
    timeout_seconds: int = 300,
) -> tuple[str, float]:
    return run_pytorch_script(
        script_content,
        gpu_type,
        eval_content,
        reference_content,
        timeout_seconds,
    )


@modal_app.function(
    gpu="a100",
    image=Image.debian_slim(python_version="3.10").pip_install(["torch"]),
)
def run_pytorch_script_a100_40gb(
    script_content: str,
    gpu_type: str,
    eval_content: str = None,
    reference_content: str = None,
    timeout_seconds: int = 300,
) -> tuple[str, float]:
    return run_pytorch_script(
        script_content,
        gpu_type,
        eval_content,
        reference_content,
        timeout_seconds,
    )


@modal_app.function(
    gpu="h100",
    image=Image.debian_slim(python_version="3.10").pip_install(["torch"]),
)
def run_pytorch_script_h100(
    script_content: str,
    gpu_type: str,
    eval_content: str = None,
    reference_content: str = None,
    timeout_seconds: int = 300,
) -> tuple[str, float]:
    return run_pytorch_script(
        script_content,
        gpu_type,
        eval_content,
        reference_content,
        timeout_seconds,
    )


def run_pytorch_script(
    script_content: str,
    gpu_type: str,
    eval_content: str = None,
    reference_content: str = None,
    timeout_seconds: int = 300,
) -> tuple[str, float]:
    """
    Executes the provided PyTorch GPU kernel in an isolated environment with a timeout

    Args:
        script_content: The PyTorch script containing the GPU kernel to benchmark
        timeout_seconds: Maximum execution time before timeout (default: 300 seconds)

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)

    NOTE: Modal execution time is not programmatically accessible, so we manually calculate it
    """
    import sys
    from io import StringIO
    import time

    # Capture stdout
    output = StringIO()
    sys.stdout = output

    try:
        with timeout(timeout_seconds):
            # Create a new dictionary for local variables to avoid polluting the global namespace

            if eval_content is not None:
                global_vars = {}
                local_vars = {}

                # I'm worried that this will create clashes in the future
                #  TODO: maybe randomized function names here?
                exec(script_content, global_vars, local_vars)
                print("Global variables after execution script:", global_vars)

                reference_content = strip_imports(reference_content, local_vars)
                exec(reference_content, global_vars, local_vars)
                print("Global variables after execution ref:", global_vars)

                eval_content = strip_imports(eval_content, local_vars)
                exec(eval_content, global_vars, local_vars)

                # # Execute the script in the isolated namespace
                # if not hasattr(eval_module, "metric"):
                #     raise ValueError(
                #         "'eval' script must define a `metric()` entry point."
                #     )
                # result = eval_module.metric()  # Execute t
                result = global_vars["metric"]()

            else:
                local_vars = {}

                execution_start_time = time.perf_counter()

                # Execute the script in the isolated namespace
                exec(script_content, {}, local_vars)

                execution_end_time = time.perf_counter()

                result = (execution_end_time - execution_start_time) * 1000

        return output.getvalue(), result

    except TimeoutException as e:
        return f"Timeout Error: {str(e)}", 0.0
    except Exception as e:
        return f"Error executing script: {str(e)}", 0.0
    finally:
        sys.stdout = sys.__stdout__


@modal_app.function(
    gpu="T4",
    image=Image.from_registry(
        "nvidia/cuda:12.6.0-devel-ubuntu24.04", add_python="3.11"
    ),
)
def run_cuda_script(
    script_content: str, timeout_seconds: int = 600
) -> tuple[str, float]:
    """
    Executes the provided CUDA kernel in an isolated environment with a timeout

    Args:
        script_content: The CUDA script containing the GPU kernel
        timeout_seconds: Maximum execution time in seconds (default: 600 seconds)

    Returns:
        tuple[str, float]: (Kernel output, execution time in milliseconds)

    NOTE: Modal execution time is not programmatically accessible, so we manually calculate it
    """
    import sys
    from io import StringIO
    import subprocess
    import os
    import time

    # Capture stdout
    output = StringIO()
    sys.stdout = output

    try:
        with timeout(timeout_seconds):
            execution_start_time = time.perf_counter()

            # Compile the CUDA code
            with open("script.cu", "w") as f:
                f.write(script_content)

            compile_process = subprocess.run(
                ["nvcc", "script.cu", "-o", "script.out"],
                capture_output=True,
                text=True,
            )

            if compile_process.returncode != 0:
                return f"Compilation Error:\n{compile_process.stderr}", 0.0

            run_process = subprocess.run(
                ["./script.out"], capture_output=True, text=True
            )
            execution_end_time = time.perf_counter()

            execution_time_sec = execution_end_time - execution_start_time
            execution_time_ms = execution_time_sec * 1000

            return run_process.stdout, execution_time_ms

    except TimeoutException as e:
        return f"Timeout Error: {str(e)}", 0.0
    except Exception as e:
        return f"Error: {str(e)}", 0.0
    finally:
        if os.path.exists("script.cu"):
            os.remove("script.cu")
        if os.path.exists("script.out"):
            os.remove("script.out")
        sys.stdout = sys.__stdout__
