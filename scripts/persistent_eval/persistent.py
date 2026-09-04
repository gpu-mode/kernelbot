"""Trusted-example-only prototype of a persistent KernelBot GPU evaluator."""

import contextlib
import gc
import importlib.util
import io
import json
import os
import runpy
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

SOCKET = "/tmp/kernelbot-persistent-evaluator.sock"


@contextlib.contextmanager
def running_server():
    """Own the experiment worker, including cleanup when a request fails."""
    Path(SOCKET).unlink(missing_ok=True)
    server = subprocess.Popen([sys.executable, str(Path(__file__).resolve())])
    try:
        deadline = time.monotonic() + 10
        while not Path(SOCKET).exists():
            if server.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Persistent evaluator failed to start")
            time.sleep(0.02)
        yield server
    finally:
        if server.poll() is None:
            try:
                request({"stop": True})
                server.wait(timeout=10)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                server.kill()
                server.wait()
        Path(SOCKET).unlink(missing_ok=True)


class DirectPool:
    """Run existing single-GPU evaluation functions in the persistent process."""

    def apply(self, function, args):
        """Use the unchanged evaluator body without starting another process."""
        return function(*args)


class ResultLogger:
    """Collect the same key/value results normally written through POPCORN_FD."""

    def __init__(self):
        """Initialize one request's output dictionary."""
        self.values = {}

    def log(self, key, value):
        """Preserve KernelBot's string result representation."""
        self.values[str(key)] = str(value)


def request(payload):
    """Send a local request to the evaluator and decode its full response."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(240)
        connection.connect(SOCKET)
        connection.sendall(json.dumps(payload).encode() + b"\n")
        with connection.makefile("rb") as handle:
            return json.loads(handle.readline())


def run_program(args, seed, timeout, multi_gpu=False, extra_env=None):
    """Replace only the execution transport for this diagnostic experiment."""
    from libkernelbot.run_eval import RunResult

    if multi_gpu:
        raise ValueError("Persistent prototype covers single-GPU trusted examples only")
    environment = {
        key: value
        for key, value in os.environ.items()
        if key
        in {
            "TORCH_EXTENSIONS_DIR",
            "TRITON_CACHE_DIR",
            "CUDA_CACHE_PATH",
            "TORCH_CUDA_ARCH_LIST",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "MAX_JOBS",
        }
    }
    response = request({"args": args, "seed": seed, "cwd": os.getcwd(), "env": environment})
    return RunResult(**response)


def evaluate(payload):
    """Load a trusted submission and use the original correctness/timing bodies."""
    args = payload["args"]
    os.chdir(payload["cwd"])
    os.environ.update(payload["env"])
    sys.path.insert(0, os.getcwd())
    for name in ("task", "utils", "reference", "submission"):
        sys.modules.pop(name, None)
    stdout, stderr = io.StringIO(), io.StringIO()
    logger = ResultLogger()
    started = time.monotonic()
    code = 0
    try:
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            if args[1] == "submission.py":
                runpy.run_path("submission.py", run_name="__main__")
            else:
                if args[2] != "benchmark":
                    raise ValueError("This prototype exercises benchmark mode only")
                spec = importlib.util.spec_from_file_location("kb_persistent_eval", Path("eval.py"))
                module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = module
                spec.loader.exec_module(module)
                module.set_seed(payload["seed"] or 42)
                tests = module.get_test_cases(args[3], payload["seed"])
                code = module.run_benchmarking(logger, DirectPool(), tests)
                del module
                sys.modules.pop(spec.name, None)
            import torch

            if torch.cuda.is_initialized():
                torch.cuda.synchronize()
                gc.collect()
                torch.cuda.empty_cache()
    except Exception:
        code = 1
        stderr.write(traceback.format_exc())
    finally:
        sys.path.pop(0)
    return {
        "success": code in (0, 112),
        "passed": logger.values.get("check") == "pass",
        "command": " ".join(args),
        "stdout": stdout.getvalue(),
        "stderr": stderr.getvalue(),
        "exit_code": code,
        "duration": time.monotonic() - started,
        "result": logger.values,
    }


def serve():
    """Serve sequential requests on one GPU, keeping Torch and CUDA initialized."""
    Path(SOCKET).unlink(missing_ok=True)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(SOCKET)
        server.listen()
        while True:
            connection, _ = server.accept()
            with connection:
                with connection.makefile("rb") as handle:
                    payload = json.loads(handle.readline())
                if payload.get("stop"):
                    connection.sendall(b"{}\n")
                    break
                result = evaluate(payload)
                connection.sendall(json.dumps(result).encode() + b"\n")
    Path(SOCKET).unlink(missing_ok=True)


if __name__ == "__main__":
    serve()
