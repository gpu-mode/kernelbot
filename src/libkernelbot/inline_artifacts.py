"""Capture load_inline libraries and reload them in a matching GPU container."""

import hashlib
import importlib.util
import inspect
import json
import os
import platform
import subprocess
import sysconfig
from pathlib import Path


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _save_module(directory: Path, module, work: Path) -> None:
    directory.mkdir(exist_ok=True)
    binary = Path(module.__file__).read_bytes()
    (directory / "extension.so").write_bytes(binary)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "module_name": module.__name__,
                "sha256": _digest(binary),
                "key": directory.name,
                "dependencies": _dependencies(Path(module.__file__).parent, work),
            },
            sort_keys=True,
        )
    )


def _load_module(directory: Path, work: Path):
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["key"] != directory.name:
        raise RuntimeError("CPU-built extension manifest does not match this request")
    _check_dependencies(manifest.get("dependencies"), work)
    binary_path = directory / "extension.so"
    if _digest(binary_path.read_bytes()) != manifest["sha256"]:
        raise RuntimeError("CPU-built extension failed SHA-256 verification")
    spec = importlib.util.spec_from_file_location(manifest["module_name"], binary_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _dependencies(build: Path, work: Path) -> dict[str, str]:
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
    dependencies = {}
    for name in sorted(paths):
        path = (build / name).resolve()
        if path.parent == build.resolve() and path.name in {"main.cpp", "cuda.cu", "sycl.sycl"}:
            continue  # Generated from the source strings already in the request key.
        name = str(path.relative_to(work)) if path.is_relative_to(work) else str(path)
        dependencies[name] = _digest(path.read_bytes())
    return dependencies


def _check_dependencies(dependencies: dict[str, str] | None, work: Path) -> None:
    if dependencies is None:
        raise RuntimeError("Artifact has no compiler dependency information")
    for name, digest in dependencies.items():
        path = work / name
        if _digest(path.read_bytes()) != digest:
            raise RuntimeError(f"Build dependency changed: {path}")


def install() -> None:
    """Install in submission interpreters, including spawned evaluators."""
    import torch
    from torch.utils import cpp_extension

    mode = os.environ["KERNELBOT_INLINE_MODE"]
    work = Path(os.environ["KERNELBOT_INLINE_WORKDIR"])
    root = work / "artifacts"
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

    def record(mode, **details):
        with (root / "events.jsonl").open("a") as log:
            log.write(json.dumps({"mode": mode, **details}) + "\n")

    def load_inline(*args, **kwargs):
        try:
            return dispatch(*args, **kwargs)
        except Exception as exc:
            if mode != "auto":
                raise
            record("fallback", reason=str(exc)[:500])
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
        # Hash before PyTorch mutates source lists to inject headers/bindings.
        key = _digest(
            json.dumps({"request": request, "runtime": current_runtime}, sort_keys=True).encode()
        )
        directory = root / key
        if mode == "capture":
            module = original(*args, **kwargs)
            _save_module(directory, module, work)
        else:
            if key not in loaded:
                loaded[key] = _load_module(directory, work)
            module = loaded[key]
            record("replay")
        return module

    cpp_extension.load_inline = load_inline


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
