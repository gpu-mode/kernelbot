"""Isolated benchmark of NVIDIA's experimental NVCC PCH build.

NVIDIA_PCH_BINARIES=/path/to/x86_64_Linux_release PYTHONPATH=src:src/runners:scripts \
  uv run modal run scripts/nvcc_pch_benchmark.py --probe-only

No production app or production PCH volume is modified.
"""

import json
import os
import statistics
from pathlib import Path

import modal

os.environ["KERNELBOT_PCH_VOLUME"] = os.environ.get(
    "KERNELBOT_HOST_PCH_BENCHMARK_VOLUME", "kernelbot-nvcc-host-pch-benchmark-cuda134-20261115"
)
if os.environ["KERNELBOT_PCH_VOLUME"] == "kernelbot-torch-pch":
    raise RuntimeError("Refusing to use the production host PCH volume for this benchmark")
from modal_pch_benchmark import CPP, CUDA, GEMM_CPP, GEMM_CUDA  # noqa: E402
from modal_runner import PCH_MOUNT, cuda_image_base, pch_volume  # noqa: E402

app = modal.App("kernelbot-nvcc-pch-benchmark")
NVCC_MOUNT = "/kernelbot-nvcc-pch"
nvcc_volume = modal.Volume.from_name(
    os.environ.get("KERNELBOT_NVCC_PCH_VOLUME", "kernelbot-nvcc-pch-benchmark-20261115"),
    create_if_missing=True,
)
image = (
    cuda_image_base.apt_install(
        "cuda-compiler-13-4", "cuda-cudart-dev-13-4", "cuda-driver-dev-13-4"
    )
    .run_commands(
        "ln -sfn /usr/local/cuda-13.4 /usr/local/cuda",
        "nvcc --version",
        "python /opt/kernelbot-pch/compiler.py --kernelbot-fingerprint",
    )
    .add_local_python_source(
        "libkernelbot", "modal_runner", "modal_runner_archs", "modal_pch_benchmark"
    )
)
if modal.is_local():
    binary_root = os.environ.get("NVIDIA_PCH_BINARIES")
    if not binary_root:
        raise RuntimeError(
            "Set NVIDIA_PCH_BINARIES to the extracted x86_64_Linux_release directory"
        )
    image = image.add_local_dir(binary_root, "/opt/nvidia-pch-binaries")

CASES = ("simple", "cutlass")
DEFAULT_MODES = ("stock_no_pch", "stock_host_pch", "patched_host_pch", "patched_all_pch")
MODES = (*DEFAULT_MODES, "patched_all_pch_prefetch")
CUDA_FLAGS = [
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "--expt-relaxed-constexpr",
    "-O3",
    "-I/opt/cutlass/include",
    "-I/opt/cutlass/tools/util/include",
]
OBSERVER = r"""#!/usr/bin/env python3
import json, os, subprocess, sys, time
kind = "nvcc" if sys.argv[0].endswith("nvcc") else "cxx"
compiler = "/usr/local/cuda/bin/nvcc" if kind == "nvcc" else os.environ["PCH_TEST_COMPILER"]
args = sys.argv[1:]
if kind == "nvcc" and os.environ.get("PCH_CAPTURE") == "1" and "-c" in args:
    with open(os.environ["PCH_TEST_COMMANDS"] + ".capture", "w") as f:
        json.dump(args, f)
    sys.exit(1)
if kind == "nvcc" and "-c" in args and os.environ.get("PCH_NVCC_ENTRY"):
    entry = os.environ["PCH_NVCC_ENTRY"]
    args = [*args, "-include", entry + "/gpumode.cuh", "--pch", "--pch-dir", entry + "/pch"]
started = time.perf_counter()
code = subprocess.call([compiler, *args])
if "-c" in args:
    with open(os.environ["PCH_TEST_COMMANDS"] + "." + kind, "a") as f:
        f.write(json.dumps({
            "args": args, "seconds": time.perf_counter()-started, "returncode": code
        }) + "\n")
sys.exit(code)
"""

LOADER = r"""
import json, os, time
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline
root = Path(os.environ["PCH_TEST_ROOT"])
case = os.environ["PCH_TEST_CASE"]
variant = int(os.environ["PCH_TEST_VARIANT"])
options = json.loads((root / "options.json").read_text())
started = time.perf_counter()
prefetch_bytes = 0
if os.environ.get("PCH_PREFETCH") == "1":
    for pch in sorted((Path(os.environ["PCH_NVCC_ENTRY"]) / "pch").glob("*.pch")):
        with pch.open("rb") as source:
            while block := source.read(16 * 1024 * 1024):
                prefetch_bytes += len(block)
    assert prefetch_bytes > 0 and prefetch_bytes == int(os.environ["PCH_EXPECTED_BYTES"]), \
        "Prefetch did not read exactly the warmed PCH payload"
prefetch_seconds = time.perf_counter() - started if prefetch_bytes else 0
compile_started = time.perf_counter()
module = load_inline(cpp_sources=(root / "cpp.txt").read_text(),
    cuda_sources=(root / "cuda.txt").read_text(), functions=["add_cuda", "version"],
    no_implicit_headers=False, verbose=True, build_directory=str(root), **options)
seconds = time.perf_counter() - started
load_inline_seconds = time.perf_counter() - compile_started
assert module.version() == variant + 1
outputs, errors = [], []
if case == "cutlass":
    torch.manual_seed(17)
    with torch.cuda.stream(torch.cuda.Stream()):
        for m, n, k in ((128, 128, 64), (129, 136, 64), (256, 256, 128), (512, 512, 256)):
            a = torch.randn(m, k, device="cuda", dtype=torch.float16) / 4
            b = torch.randn(n, k, device="cuda", dtype=torch.float16) / 4
            for zero in (False, True):
                if zero:
                    a.zero_()
                actual = module.add_cuda(a, b)
                reference = (a.double() @ b.double().T) * (variant + 1)
                torch.testing.assert_close(actual.double(), reference, rtol=2e-4, atol=2e-4)
                errors.append(float((actual.double() - reference).abs().max()))
        a = torch.ones(8, 8, device="cuda", dtype=torch.float16)
        actual = module.add_cuda(a, a)
        torch.testing.assert_close(
            actual, torch.full_like(actual, 8 * (variant + 1)), rtol=0, atol=0
        )
        outputs.append(float(actual[0, 0]))
else:
    for n in (1, 127, 128, 129, 1025):
        a = torch.arange(n, device="cuda", dtype=torch.float16) % 8
        b = torch.ones_like(a)
        actual = module.add_cuda(a, b)
        torch.testing.assert_close(actual, a + b + (variant + 1), rtol=0, atol=0)
        outputs.append(float(actual[0]))
(root / "result.json").write_text(json.dumps(dict(
    seconds=seconds, load_inline_seconds=load_inline_seconds,
    prefetch_seconds=prefetch_seconds, prefetch_bytes=prefetch_bytes, outputs=outputs, errors=errors
)))
"""


def prepare_build(root, case, variant, name, host_pch, entry=None, capture=False):
    cpp = GEMM_CPP if case == "cutlass" else CPP
    cuda = (
        GEMM_CUDA.replace("TILE_N", str(128 if variant == 0 else 64))
        if case == "cutlass"
        else CUDA.replace("THREADS", str(256 if variant == 0 else 128))
    )
    header = f"#pragma once\nconstexpr int kVariant = {variant + 1};\n"
    (root / "variant.h").write_text(header)
    (root / "cpp.txt").write_text(cpp)
    (root / "cuda.txt").write_text(cuda)
    (root / "options.json").write_text(
        json.dumps(
            {
                "name": name,
                "extra_cuda_cflags": CUDA_FLAGS if case == "cutlass" else [],
            }
        )
    )
    for kind in ("cxx", "nvcc"):
        p = root / f"observe-{kind}"
        p.write_text(OBSERVER)
        p.chmod(0o755)
    env = dict(os.environ)
    for key in ("MAX_JOBS", "KERNELBOT_PCH_DISABLE", "KERNELBOT_PCH_TRACE", "KERNELBOT_PCH_WRITE"):
        env.pop(key, None)
    env.update(
        {
            "TORCH_CUDA_ARCH_LIST": "7.5",
            "CXX": str(root / "observe-cxx"),
            "PYTORCH_NVCC": str(root / "observe-nvcc"),
            "PCH_TEST_COMPILER": "/opt/kernelbot-pch/compiler.py" if host_pch else "/usr/bin/g++",
            "PCH_TEST_ROOT": str(root),
            "PCH_TEST_CASE": case,
            "PCH_TEST_VARIANT": str(variant),
            "PCH_TEST_COMMANDS": str(root / "commands"),
            "PCH_CAPTURE": "1" if capture else "0",
            "PCH_NVCC_ENTRY": str(entry) if entry else "",
        }
    )
    return env, cpp + cuda + header


def pch_identity(case, patched):
    header = (
        "#pragma once\n#include <ATen/ATen.h>\n#include <torch/types.h>\n"
        "#include <cuda.h>\n#include <cuda_runtime.h>\n"
    )
    header += (
        "#include <cutlass/gemm/device/gemm.h>\n"
        if case == "cutlass"
        else "#include <cuda_fp16.h>\n"
    )
    header += "#pragma hdrstop\n"
    return {
        "patch": patched,
        "fingerprint": Path("/opt/kernelbot-pch/fingerprint").read_text().strip(),
        "header": header,
        "case": case,
        "cuda_flags": CUDA_FLAGS if case == "cutlass" else [],
        "arch": "7.5",
        "protocol": 2,
    }


def install_patch():
    """Overlay only the four supplied binaries inside this disposable container."""
    import hashlib
    import re
    import shutil

    rows = []
    for source_name, target_name in (
        ("nvcc", "bin/nvcc"),
        ("cudafe++", "bin/cudafe++"),
        ("nvvm/bin/cicc", "nvvm/bin/cicc"),
        ("nvvm-next/bin/cicc", "nvvm-next/bin/cicc"),
    ):
        source = Path("/opt/nvidia-pch-binaries") / source_name
        target = Path("/usr/local/cuda") / target_name
        existed = target.exists()
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        target.chmod(0o755)
        rows.append(
            {
                "file": str(target),
                "replaced_existing": existed,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            }
        )
    profile_path = Path("/usr/local/cuda/bin/nvcc.profile")
    # The experiment uses SM75 and SM100, both on NVIDIA's newer backend.
    # General production dispatch must also handle SM70 and earlier separately.
    profile, count = re.subn(
        r"^CICC_PATH\s*=.*$",
        "CICC_PATH        = $(TOP)/nvvm-next/bin",
        profile_path.read_text(),
        flags=re.MULTILINE,
    )
    assert count == 1, "Expected exactly one CICC_PATH assignment in CUDA's profile"
    profile_path.write_text(profile)
    rows.append(
        {
            "file": str(profile_path),
            "replaced_existing": True,
            "sha256": hashlib.sha256(profile.encode()).hexdigest(),
        }
    )
    return rows


@app.function(image=image, cpu=4, memory=16384, timeout=300, single_use_containers=True)
def probe():
    import shutil
    import subprocess
    import tempfile

    stock_version = subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True)
    paths = subprocess.check_output(
        ["find", "/usr/local/cuda/", "-name", "cicc", "-o", "-name", "nvcc.profile"], text=True
    )
    patched = install_patch()
    version = subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True)
    help_text = subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--help"], text=True)
    lines = help_text.splitlines()
    indices = {
        j
        for i, line in enumerate(lines)
        if "pch" in line.lower()
        for j in range(max(0, i - 2), min(len(lines), i + 14))
    }
    profiles = {str(p): p.read_text() for p in Path("/usr/local/cuda/bin").glob("*.profile")}
    smoke = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        source = root / "smoke.cu"
        source.write_text("__global__ void kernelbot_nvcc_smoke() {}\n")
        for arch in ("sm_75", "sm_100"):
            output = root / f"{arch}.o"
            command = [
                "/usr/local/cuda/bin/nvcc",
                f"-arch={arch}",
                "-c",
                str(source),
                "-o",
                str(output),
            ]
            dry = subprocess.run([*command, "--dryrun"], capture_output=True, text=True, timeout=60)
            compile_result = subprocess.run(
                [*command, "-v"], capture_output=True, text=True, timeout=60
            )
            dispatched = "nvvm-next/bin" in dry.stderr and "$(_NVVM_BRANCH_)" not in dry.stderr
            smoke.append(
                {
                    "arch": arch,
                    "command": command,
                    "dryrun_returncode": dry.returncode,
                    "dryrun": dry.stdout + dry.stderr,
                    "next_backend_selected": dispatched,
                    "returncode": compile_result.returncode,
                    "log": compile_result.stdout + compile_result.stderr,
                    "object_bytes": output.stat().st_size if output.exists() else 0,
                }
            )
    return {
        "stock_version": stock_version,
        "patched_version": version,
        "paths": paths,
        "patch": patched,
        "nvcc_profiles": profiles,
        "pch_help": "\n".join(lines[i] for i in sorted(indices)),
        "strace": shutil.which("strace"),
        "smoke": smoke,
        "compatible": all(
            r["returncode"] == 0 and r["next_backend_selected"] and r["object_bytes"] for r in smoke
        ),
    }


@app.function(
    image=image,
    cpu=4,
    memory=16384,
    timeout=1800,
    single_use_containers=True,
    max_containers=1,
    volumes={PCH_MOUNT: pch_volume, NVCC_MOUNT: nvcc_volume},
)
def warm(cases: list[str]):
    import hashlib
    import subprocess
    import tempfile
    import time

    patched = install_patch()
    pch_volume.reload()
    nvcc_volume.reload()
    print("Warming host CUDA-linked default profile", flush=True)
    host_started = time.perf_counter()
    host = subprocess.run(
        ["python3", "/opt/kernelbot-pch/warm.py", "cuda-default"],
        capture_output=True,
        text=True,
        timeout=300,
    )
    if host.returncode:
        raise RuntimeError(host.stdout + host.stderr)
    host_seconds = time.perf_counter() - host_started
    pch_volume.commit()
    print(f"Host profile ready in {host_seconds:.2f}s", flush=True)
    rows = []
    for case in cases:
        identity = pch_identity(case, patched)
        key = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        entry = Path(NVCC_MOUNT) / key
        entry.mkdir(parents=True, exist_ok=True)
        (entry / "gpumode.cuh").write_text(identity["header"])
        (entry / "pch").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            print(f"Capturing exact NVCC flags for {case}", flush=True)
            env, _ = prepare_build(root, case, 0, "kb_nvcc_pch_warmer", True, capture=True)
            captured = subprocess.run(
                ["python3", "-c", LOADER], env=env, capture_output=True, text=True, timeout=180
            )
            capture_path = root / "commands.capture"
            if not capture_path.exists():
                raise RuntimeError(
                    "Failed to capture NVCC command: " + captured.stdout + captured.stderr
                )
            args = json.loads(capture_path.read_text())
            flags = []
            i = 0
            while i < len(args):
                if args[i] in {"-c", "-o", "--dependency-output", "-MF", "-MT", "-MQ"}:
                    i += 2
                    continue
                if args[i] not in {"--generate-dependencies-with-compile", "-MMD", "-MD"}:
                    flags.append(args[i])
                i += 1
            command = [
                "/usr/local/cuda/bin/nvcc",
                *flags,
                "--pch-dir",
                str(entry / "pch"),
                "--pch",
                "--pch-verbose",
                "-x",
                "cu",
                "-c",
                str(entry / "gpumode.cuh"),
                "-o",
                str(root / "warm.o"),
            ]
            print(f"Creating {case} PCH in {entry}", flush=True)
            tick = time.perf_counter()
            result = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, timeout=300
            )
            seconds = time.perf_counter() - tick
            row = {
                "case": case,
                "key": key,
                "command": command,
                "seconds": seconds,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "files": [
                    {"path": str(p.relative_to(entry)), "bytes": p.stat().st_size}
                    for p in sorted((entry / "pch").rglob("*"))
                    if p.is_file()
                ],
            }
            rows.append(row)
            print(
                f"{case} warmup returned {result.returncode} after {seconds:.2f}s; "
                f"{len(row['files'])} files",
                flush=True,
            )
            if result.returncode or not row["files"]:
                raise RuntimeError(json.dumps(row))
            (entry / "metadata.json").write_text(
                json.dumps({"identity": identity, "warmup": row}, indent=2)
            )
            (Path(NVCC_MOUNT) / f"{case}.json").write_text(json.dumps({"key": key}))
            nvcc_volume.commit()
    return {"host_seconds": host_seconds, "host_log": host.stdout + host.stderr, "device": rows}


@app.function(
    image=image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=900,
    max_containers=4,
    single_use_containers=True,
    restrict_modal_access=True,
    volumes={
        PCH_MOUNT: pch_volume.with_mount_options(read_only=True),
        NVCC_MOUNT: nvcc_volume.with_mount_options(read_only=True),
    },
)
def trial(case: str, variant: int, mode: str, repeat: int):
    import hashlib
    import re
    import subprocess
    import tempfile
    import time

    import torch

    started = time.perf_counter()
    assert mode in MODES and case in CASES
    patched = install_patch() if mode.startswith("patched") else None
    nvcc_version = subprocess.check_output(["/usr/local/cuda/bin/nvcc", "--version"], text=True)
    entry = None
    if mode.startswith("patched_all_pch"):
        key = json.loads((Path(NVCC_MOUNT) / f"{case}.json").read_text())["key"]
        entry = Path(NVCC_MOUNT) / key
        expected = pch_identity(case, patched)
        metadata = json.loads((entry / "metadata.json").read_text())
        assert metadata["identity"] == expected, "Warm PCH identity does not match; rerun warmup"
        assert (entry / "gpumode.cuh").read_text() == expected["header"]
        assert key == hashlib.sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        name = f"kb_nvcc_{case}_{variant}"
        env, source = prepare_build(root, case, variant, name, mode != "stock_no_pch", entry)
        env["PCH_PREFETCH"] = "1" if mode == "patched_all_pch_prefetch" else "0"
        if entry:
            env["PCH_EXPECTED_BYTES"] = str(
                sum(f["bytes"] for f in metadata["warmup"]["files"] if f["path"].endswith(".pch"))
            )
        run_started = time.perf_counter()
        result = subprocess.run(
            ["python3", "-c", LOADER], env=env, capture_output=True, text=True, timeout=300
        )
        process_seconds = time.perf_counter() - run_started
        log = result.stdout + result.stderr
        row = {
            "case": case,
            "variant": variant,
            "mode": mode,
            "repeat": repeat,
            "extension_name": name,
            "returncode": result.returncode,
            "log": log,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "container_id": os.environ["MODAL_TASK_ID"],
            "image_id": os.environ["MODAL_IMAGE_ID"],
            "gpu": torch.cuda.get_device_name(),
            "torch": str(torch.__version__),
            "nvcc_version": nvcc_version,
            "patch": patched,
            "pch_entry": str(entry) if entry else None,
            "process_seconds": process_seconds,
        }
        if entry:
            timed_used = re.findall(r'using precompiled header file "([^"]+)"', log)
            timed_created = re.findall(r'creating precompiled header file "([^"]+)"', log)
            row["timed_device_pch_files_used"] = timed_used
            row["timed_device_pch_consumed"] = (
                result.returncode == 0
                and not timed_created
                and {Path(p).name.split("_", 1)[0] for p in timed_used} == {"cicc", "cudafe"}
                and all(Path(p).is_relative_to(entry / "pch") for p in timed_used)
            )
        for kind in ("cxx", "nvcc"):
            path = root / f"commands.{kind}"
            row[f"{kind}_commands"] = (
                [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
            )
        if result.returncode:
            return row
        measurement = json.loads((root / "result.json").read_text())
        row.update(
            {
                "seconds": measurement["seconds"],
                "outputs": measurement["outputs"],
                "load_inline_seconds": measurement["load_inline_seconds"],
                "prefetch_seconds": measurement["prefetch_seconds"],
                "prefetch_bytes": measurement["prefetch_bytes"],
                "max_absolute_error": max(measurement["errors"], default=0),
                "correctness_checks": 10 if case == "cutlass" else 6,
            }
        )
        # GCC -H prints ! for a valid consumed PCH. Keep this compilation untimed.
        cxx_args = list(row["cxx_commands"][0]["args"])
        cxx_args[cxx_args.index("-o") + 1] = str(root / "diagnostic.o")
        diagnostic = subprocess.run(
            [env["PCH_TEST_COMPILER"], *cxx_args, "-H"],
            env=env,
            cwd=root,
            capture_output=True,
            text=True,
            timeout=180,
        )
        row["host_pch_consumed"] = "! /kernelbot-pch/" in diagnostic.stderr
        assert diagnostic.returncode == 0, diagnostic.stderr[-6000:]
        assert row["host_pch_consumed"] == (mode != "stock_no_pch"), diagnostic.stderr[-6000:]
        row["host_hit_keys"] = re.findall(r"\[kernelbot-pch\] hit (\w+)", log)
        # Preserve verbose NVCC diagnostics independently of the measured build.
        if mode.startswith("patched"):
            args = list(row["nvcc_commands"][0]["args"])
            args[args.index("-o") + 1] = str(root / "diagnostic.cuda.o")
            phase_file = root / "nvcc-diagnostic.csv"
            extra = ["-v", "--time", str(phase_file), *(["--pch-verbose"] if entry else [])]
            diagnostic = subprocess.run(
                ["/usr/local/cuda/bin/nvcc", *args, *extra],
                env=env,
                cwd=root,
                capture_output=True,
                text=True,
                timeout=180,
            )
            row["nvcc_diagnostic"] = {
                "returncode": diagnostic.returncode,
                "stdout": diagnostic.stdout,
                "stderr": diagnostic.stderr,
                "phase_csv": phase_file.read_text() if phase_file.exists() else None,
            }
            used = re.findall(r'using precompiled header file "([^"]+)"', diagnostic.stderr)
            created = re.findall(r'creating precompiled header file "([^"]+)"', diagnostic.stderr)
            row["device_pch_files_used"] = used
            row["device_pch_files_created"] = created
            frontends = {Path(path).name.split("_", 1)[0] for path in used}
            row["device_pch_consumed"] = bool(entry) and (
                frontends == {"cicc", "cudafe"}
                and not created
                and diagnostic.returncode == 0
                and all(Path(path).is_relative_to(entry / "pch") for path in used)
            )
        row["remote_seconds"] = time.perf_counter() - started
        return row


@app.local_entrypoint()
def main(
    output: str = "/tmp/kernelbot-nvcc-pch.json",
    probe_only: bool = False,
    warm_only: bool = False,
    skip_warm: bool = False,
    cases: str = "simple,cutlass",
    modes: str = ",".join(DEFAULT_MODES),
    repeats: int = 2,
):
    selected = cases.split(",")
    selected_modes = modes.split(",")
    assert set(selected) <= set(CASES) and set(selected_modes) <= set(MODES)
    assert repeats >= 1
    result = {"probe": probe.remote(), "rows": []}

    def save():
        Path(output).write_text(json.dumps(result, indent=2) + "\n")

    save()
    if probe_only:
        print(json.dumps(result["probe"], indent=2))
        return
    assert result["probe"]["compatible"], (
        f"Patched compiler compatibility checks failed; inspect {output}"
    )
    if not skip_warm:
        result["warmup"] = warm.remote(selected)
        save()
        print(json.dumps(result["warmup"], indent=2))
    if warm_only:
        return
    jobs = [
        (case, variant, mode, repeat)
        for repeat in range(repeats)
        for case in selected
        for variant in (0, 1)
        for mode in (selected_modes if (variant + repeat) % 2 == 0 else selected_modes[::-1])
    ]
    for row in trial.starmap(jobs):
        result["rows"].append(row)
        save()
        print(
            json.dumps(
                {
                    k: row.get(k)
                    for k in ("case", "variant", "mode", "repeat", "returncode", "seconds")
                }
            )
        )
    assert len({r["container_id"] for r in result["rows"]}) == len(result["rows"])
    for case in selected:
        for mode in selected_modes:
            group = [r for r in result["rows"] if r["case"] == case and r["mode"] == mode]
            good = [r["seconds"] for r in group if r["returncode"] == 0]
            print(
                f"{case}/{mode}: median={statistics.median(good) if good else None}; "
                f"passed={len(good)}/{len(group)}"
            )
    assert all(row["returncode"] == 0 for row in result["rows"]), (
        f"Failed builds retained in {output}"
    )
    assert all(
        row.get("device_pch_consumed")
        for row in result["rows"]
        if row["mode"].startswith("patched_all_pch")
    ), f"NVCC PCH consumption was not verified; inspect diagnostics in {output}"
    assert all(
        row.get("timed_device_pch_consumed")
        for row in result["rows"]
        if row["mode"].startswith("patched_all_pch")
    ), f"NVCC PCH consumption was not verified in the timed build; inspect logs in {output}"
