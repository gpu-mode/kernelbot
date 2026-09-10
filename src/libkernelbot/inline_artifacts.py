"""Capture load_inline libraries and reload them in a matching GPU container."""

import hashlib
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sysconfig
import time
from pathlib import Path


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _save_module(directory: Path, module, request: dict, runtime: dict, work: Path) -> None:
    directory.mkdir(exist_ok=True)
    binary = Path(module.__file__).read_bytes()
    (directory / "extension.so").write_bytes(binary)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "module_name": module.__name__,
                "sha256": _digest(binary),
                "runtime": runtime,
                "request": request,
                "dependencies": _dependencies(Path(module.__file__).parent, work),
            },
            sort_keys=True,
        )
    )


def _load_module(directory: Path, request: dict, runtime: dict, work: Path):
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"No CPU-built artifact for {request['name']!r}")
    manifest = json.loads(manifest_path.read_text())
    if manifest["runtime"] != runtime or manifest["request"] != request:
        raise RuntimeError("CPU-built extension manifest does not match this request")
    _check_dependencies(manifest.get("dependencies"), work)
    binary_path = directory / "extension.so"
    if _digest(binary_path.read_bytes()) != manifest["sha256"]:
        raise RuntimeError("CPU-built extension failed SHA-256 verification")
    spec = importlib.util.spec_from_file_location(manifest["module_name"], binary_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _refuse_build(*args, **kwargs):
    raise RuntimeError("GPU compilation is disabled in artifact replay mode")


def _dependencies(build: Path, work: Path) -> list[dict]:
    result = subprocess.run(
        ["ninja", "-C", str(build), "-t", "deps"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    paths = {line.strip() for line in result.stdout.splitlines() if line.startswith("    ")}
    if not paths:
        raise RuntimeError("Compiler dependency information is unavailable")
    work = work.resolve()
    dependencies = []
    for name in sorted(paths):
        path = (build / name).resolve()
        if path.parent == build.resolve() and path.name in {"main.cpp", "cuda.cu", "sycl.sycl"}:
            continue  # Generated from the source strings already in the request key.
        relative = path.is_relative_to(work)
        dependencies.append(
            {
                "path": str(path.relative_to(work)) if relative else str(path),
                "relative": relative,
                "sha256": _digest(path.read_bytes()),
            }
        )
    return dependencies


def _check_dependencies(dependencies: list[dict] | None, work: Path) -> None:
    if dependencies is None:
        raise RuntimeError("Artifact has no compiler dependency information")
    for dependency in dependencies:
        path = Path(dependency["path"])
        if dependency["relative"]:
            path = work / path
        if _digest(path.read_bytes()) != dependency["sha256"]:
            raise RuntimeError(f"Build dependency changed: {path}")


def install() -> None:
    """Install in submission interpreters, including spawned evaluators."""
    import torch
    from torch.utils import cpp_extension

    mode = os.environ["KERNELBOT_INLINE_MODE"]
    if mode not in {"capture", "replay", "auto"}:
        raise ValueError(f"Unknown inline artifact mode: {mode}")
    root = Path(os.environ["KERNELBOT_INLINE_ARTIFACTS"])
    work = Path(os.environ["KERNELBOT_INLINE_WORKDIR"])
    root.mkdir(parents=True, exist_ok=True)
    original = cpp_extension.load_inline
    signature = inspect.signature(original)
    runtime = {
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "python_abi": sysconfig.get_config_var("SOABI"),
        "machine": platform.machine(),
        "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
    }
    loaded = {}

    def load_inline(*args, **kwargs):
        try:
            return dispatch(*args, **kwargs)
        except Exception as exc:
            if mode != "auto":
                raise
            with (root / "events.jsonl").open("a") as log:
                log.write(json.dumps({"mode": "fallback", "reason": str(exc)[:500]}) + "\n")
            return original(*args, **kwargs)

    def dispatch(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not bound.arguments.get("is_python_module", True) or bound.arguments.get(
            "extra_ldflags"
        ):
            raise ValueError("Non-Python modules and custom linker inputs require GPU compilation")
        # Destination paths and logging do not affect the binary's identity.
        request = {
            k: v for k, v in bound.arguments.items() if k not in {"build_directory", "verbose"}
        }
        # PyTorch may mutate source lists while injecting headers/bindings.
        request = json.loads(json.dumps(request))
        current_runtime = {
            **runtime,
            "cuda_home": getattr(cpp_extension, "CUDA_HOME", None),
            "build_environment": {
                name: os.environ.get(name)
                for name in (
                    "TORCH_CUDA_ARCH_LIST",
                    "CXX",
                    "CC",
                    "CUDA_HOME",
                    "CPATH",
                    "CPLUS_INCLUDE_PATH",
                    "C_INCLUDE_PATH",
                    "NVCC_PREPEND_FLAGS",
                    "NVCC_APPEND_FLAGS",
                    "LIBRARY_PATH",
                    "LD_LIBRARY_PATH",
                )
            },
        }
        key = _digest(
            json.dumps({"request": request, "runtime": current_runtime}, sort_keys=True).encode()
        )
        directory = root / key
        started = time.perf_counter()
        if mode == "capture":
            module = original(*args, **kwargs)
            elapsed = time.perf_counter() - started
            _save_module(directory, module, request, current_runtime, work)
        else:
            if key not in loaded:
                loaded[key] = _load_module(directory, request, current_runtime, work)
            module = loaded[key]
            elapsed = time.perf_counter() - started
        # One append per event also works for the evaluator's spawned processes.
        with (root / "events.jsonl").open("a") as log:
            log.write(
                json.dumps(
                    {
                        "mode": "replay" if mode == "auto" else mode,
                        "key": key,
                        "seconds": elapsed,
                        "pid": os.getpid(),
                    }
                )
                + "\n"
            )
        return module

    cpp_extension.load_inline = load_inline
    if mode == "replay":
        cpp_extension._run_ninja_build = _refuse_build


def prepare_environment(work: Path, mode: str, arch: str) -> dict[str, str]:
    """Configure the hook in child interpreters without patching the runner."""
    hook = work / "bootstrap"
    hook.mkdir(exist_ok=True)
    (hook / "sitecustomize.py").write_text(
        "import os, traceback\n"
        "try:\n"
        "    from libkernelbot.inline_artifacts import install\n"
        "    install()\n"
        "except BaseException:\n"
        "    traceback.print_exc()\n"
        "    os._exit(1)\n"
    )
    return {
        **os.environ,
        "KERNELBOT_INLINE_MODE": mode,
        "KERNELBOT_INLINE_WORKDIR": str(work),
        "KERNELBOT_INLINE_ARTIFACTS": str(work / "artifacts"),
        "TORCH_EXTENSIONS_DIR": str(work / "build"),
        "TORCH_CUDA_ARCH_LIST": arch,
        "MAX_JOBS": "2",
        "PYTHONPATH": os.pathsep.join(
            filter(
                None,
                [
                    str(hook),
                    str(work),
                    str(Path(__file__).resolve().parents[1]),
                    os.environ.get("PYTHONPATH", ""),
                ],
            )
        ),
    }


def pack_artifacts(root: Path) -> dict[str, bytes]:
    """Transfer only the libraries and manifests, never the Ninja build cache."""
    return {
        str(path.relative_to(root)): path.read_bytes()
        for pattern in ("*/manifest.json", "*/extension.so")
        for path in root.glob(pattern)
    }


def unpack_artifacts(root: Path, files: dict[str, bytes]) -> None:
    for name, data in files.items():
        path = root / name
        if not path.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Invalid artifact path: {name}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def read_events(root: Path) -> list[dict]:
    path = root / "events.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
