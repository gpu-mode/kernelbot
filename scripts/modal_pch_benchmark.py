"""Benchmark Torch PCH reuse with CUDA, CUTLASS, file-based builds, and cache misses.

PYTHONPATH=src:src/runners KERNELBOT_PCH_VOLUME=kernelbot-pch-test \
    uv run modal run scripts/modal_pch_benchmark.py
"""

import json
import statistics
from pathlib import Path

import modal
from modal_runner import PCH_MOUNT, cuda_image, pch_volume, warm_pch

app = modal.App("kernelbot-pch-matrix")
app.include(warm_pch.app)
CASES = (
    "inline_simple", "load_files", "plain_cpp", "flag_miss",
    "cutlass_gemm", "cutlass_gemm_implicit",
)

CACHED_CASES = set(CASES) - {"plain_cpp", "flag_miss"}

# Instrument both modes identically; do not enable GCC's verbose header tracing
# in timed builds. Baseline delegates directly to g++, cache mode to the wrapper.
OBSERVER = r'''#!/usr/bin/env python3
import json, os, subprocess, sys, time
started = time.perf_counter()
kind = "nvcc" if sys.argv[0].endswith("nvcc") else "cxx"
compiler = "/usr/local/cuda/bin/nvcc" if kind == "nvcc" else os.environ["PCH_TEST_COMPILER"]
code = subprocess.call([compiler, *sys.argv[1:]])
if "-c" in sys.argv:
    with open(os.environ["PCH_TEST_COMMANDS"] + "." + kind, "a") as f:
        f.write(json.dumps({"args": sys.argv[1:], "seconds": time.perf_counter()-started}) + "\n")
sys.exit(code)
'''

CPP = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "variant.h"

void add_impl(const void*, const void*, void*, int, cudaStream_t);
int version() { return kVariant; }
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device() == b.device());
    TORCH_CHECK(a.scalar_type() == torch::kFloat16 && b.scalar_type() == torch::kFloat16);
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous() && a.sizes() == b.sizes());
    c10::cuda::CUDAGuard guard(a.device());
    auto out = torch::empty_like(a);
    add_impl(a.data_ptr(), b.data_ptr(), out.data_ptr(), a.numel(),
             c10::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess);
    return out;
}
'''

CUDA = r'''
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "variant.h"
__global__ void add_kernel(const __half* a, const __half* b, __half* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __hadd(__hadd(a[i], b[i]), __float2half(float(kVariant)));
}
void add_impl(const void* a, const void* b, void* out, int n, cudaStream_t stream) {
    constexpr int threads = THREADS;
    add_kernel<<<(n + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __half*>(a), static_cast<const __half*>(b), static_cast<__half*>(out), n);
}
'''

GEMM_CPP = r'''
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "variant.h"

int gemm_impl(const void*, const void*, float*, int, int, int, cudaStream_t);
int version() { return kVariant; }
torch::Tensor add_cuda(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.is_cuda() && b.is_cuda() && a.device() == b.device());
    TORCH_CHECK(a.scalar_type() == torch::kFloat16 && b.scalar_type() == torch::kFloat16);
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2 && a.is_contiguous() && b.is_contiguous());
    TORCH_CHECK(a.size(1) == b.size(1) && a.size(1) % 8 == 0 && b.size(0) % 4 == 0);
    TORCH_CHECK(a.size(0) > 0 && b.size(0) > 0 && a.size(1) > 0);
    TORCH_CHECK(a.size(0) < INT_MAX && b.size(0) < INT_MAX && a.size(1) < INT_MAX);
    c10::cuda::CUDAGuard guard(a.device());
    auto out = torch::empty({a.size(0), b.size(0)}, a.options().dtype(torch::kFloat32));
    int status = gemm_impl(a.data_ptr(), b.data_ptr(), out.data_ptr<float>(),
                          a.size(0), b.size(0), a.size(1),
                          c10::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(status == 0, "CUTLASS status: ", status);
    return out;
}
'''

GEMM_CUDA = r'''
#include <cuda_runtime.h>
#include <cutlass/gemm/device/gemm.h>
#include "variant.h"

// A[M,K] and B[N,K] are contiguous: interpret B as column-major [K,N].
using Gemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor, float,
    cutlass::arch::OpClassTensorOp, cutlass::arch::Sm75,
    cutlass::gemm::GemmShape<128, TILE_N, 32>,
    cutlass::gemm::GemmShape<64, 64, 32>,
    cutlass::gemm::GemmShape<16, 8, 8>,
    cutlass::epilogue::thread::LinearCombination<float, 4, float, float>,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, 2>;

int gemm_impl(const void* a, const void* b, float* out, int m, int n, int k, cudaStream_t stream) {
    Gemm::Arguments args({m, n, k},
        {static_cast<const cutlass::half_t*>(a), k},
        {static_cast<const cutlass::half_t*>(b), k},
        {out, n}, {out, n}, {float(kVariant), 0.0f});
    auto status = Gemm::can_implement(args);
    if (status != cutlass::Status::kSuccess) return int(status);
    if (Gemm::get_workspace_size(args) != 0) return int(cutlass::Status::kErrorInvalidProblem);
    return int(Gemm{}(args, nullptr, stream));
}
'''


@app.function(
    image=cuda_image,
    gpu="T4",
    cpu=4,
    memory=16384,
    timeout=900,
    max_containers=4,
    single_use_containers=True,
    restrict_modal_access=True,
    volumes={PCH_MOUNT: pch_volume.with_mount_options(read_only=True)},
)
def trial(case: str, variant: int, enabled: bool, repeat: int = 0):
    import hashlib
    import os
    import re
    import subprocess
    import tempfile
    import time

    import torch

    started = time.perf_counter()
    os.environ.pop("MAX_JOBS", None)
    os.environ.pop("KERNELBOT_PCH_DISABLE", None)
    os.environ.pop("KERNELBOT_PCH_TRACE", None)
    os.environ.pop("KERNELBOT_PCH_WRITE", None)
    os.environ["TORCH_CUDA_ARCH_LIST"] = "7.5"
    os.environ["PCH_TEST_COMPILER"] = "/opt/kernelbot-pch/compiler.py" if enabled else "/usr/bin/g++"
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        observer = root / "observe-cxx"
        observer.write_text(OBSERVER)
        observer.chmod(0o755)
        nvcc = root / "observe-nvcc"
        nvcc.write_text(OBSERVER)
        nvcc.chmod(0o755)
        os.environ["PYTORCH_NVCC"] = str(nvcc)
        commands = root / "commands.jsonl"
        os.environ["CXX"] = str(observer)
        os.environ["PCH_TEST_COMMANDS"] = str(commands)
        header = f"#pragma once\nconstexpr int kVariant = {variant + 1};\n"
        (root / "variant.h").write_text(header)
        cpp = CPP
        cuda = CUDA.replace("THREADS", str(256 if variant == 0 else 128))
        if case.startswith("cutlass_gemm"):
            cpp = GEMM_CPP
            cuda = GEMM_CUDA.replace("TILE_N", str(128 if variant == 0 else 64))
        name = "pch_matrix_shared" if case == "load_files" else f"pch_matrix_{case}_{variant}_{int(enabled)}"
        if case == "plain_cpp":
            cpp = '#include <array>\n#include <cstdio>\n#include "variant.h"\nint main() {\n'
            cpp += 'std::array<int, 3> a{1, 2, 3}; std::printf("%d\\n", a[1] + kVariant); }\n'
            (root / "main.cpp").write_text(cpp)
            # Multiple direct compilations resolve wrapper overhead at millisecond scale.
            times = []
            logs = []
            for _ in range(10):
                tick = time.perf_counter()
                result = subprocess.run(
                    [str(observer), "-std=c++20", "-c", str(root / "main.cpp"), "-o", str(root / "main.o")],
                    capture_output=True, text=True, check=True,
                )
                times.append(time.perf_counter() - tick)
                logs.append(result.stdout + result.stderr)
            subprocess.run(["g++", str(root / "main.o"), "-o", str(root / "main")], check=True)
            actual = int(subprocess.check_output([str(root / "main")]))
            assert actual == variant + 3
            wall = statistics.median(times)
            log = "\n".join(logs)
            correctness_checks = 1
            device_outputs = [actual]
        else:
            (root / "binding.cpp").write_text(
                cpp + '\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n'
                'm.def("add_cuda", &add_cuda); m.def("version", &version); }\n'
            )
            (root / "kernel.cu").write_text(cuda)
            (root / "cpp.txt").write_text(cpp)
            (root / "cuda.txt").write_text(cuda)
            # A subprocess captures compiler output and limits timed work to load/load_inline.
            loader = '''
import json, os, time
from pathlib import Path
import torch
from torch.utils.cpp_extension import load, load_inline
root = Path(os.environ["PCH_TEST_ROOT"])
case = os.environ["PCH_TEST_CASE"]
variant = int(os.environ["PCH_TEST_VARIANT"])
options = dict(name=os.environ["PCH_TEST_NAME"], build_directory=str(root), verbose=True)
if case.startswith("cutlass_gemm"):
    # Restore CUDA half operators for CUTLASS; host flags stay unchanged.
    options["extra_cuda_cflags"] = [
        "-U__CUDA_NO_HALF_OPERATORS__", "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__", "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "-O3", "-I/opt/cutlass/include", "-I/opt/cutlass/tools/util/include",
    ]
started = time.perf_counter()
if case == "load_files":
    module = load(sources=[str(root / "binding.cpp"), str(root / "kernel.cu")], **options)
else:
    module = load_inline(cpp_sources=(root / "cpp.txt").read_text(), cuda_sources=(root / "cuda.txt").read_text(),
        functions=["add_cuda", "version"], no_implicit_headers=case != "cutlass_gemm_implicit",
        extra_cflags=[f"-DPCH_CASE={variant+1}"] if case == "flag_miss" else [], **options)
seconds = time.perf_counter() - started
assert module.version() == variant + 1
outputs = []
errors = []
if case.startswith("cutlass_gemm"):
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
        torch.testing.assert_close(actual, torch.full_like(actual, 8 * (variant + 1)), rtol=0, atol=0)
        outputs.append(float(actual[0, 0]))
else:
    for n in (1, 127, 128, 129, 1025):
        a = torch.arange(n, device="cuda", dtype=torch.float16) % 8
        b = torch.ones_like(a)
        actual = module.add_cuda(a, b)
        expected = a + b + (variant + 1)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        outputs.append(float(actual[0]))
(root / "result.json").write_text(json.dumps(dict(seconds=seconds, outputs=outputs, errors=errors)))
'''
            env = os.environ | {
                "PCH_TEST_ROOT": str(root), "PCH_TEST_CASE": case,
                "PCH_TEST_VARIANT": str(variant), "PCH_TEST_NAME": name,
            }
            result = subprocess.run(["python3", "-c", loader], env=env, capture_output=True, text=True)
            log = result.stdout + result.stderr
            assert result.returncode == 0, log[-14000:]
            measurement = json.loads((root / "result.json").read_text())
            wall = measurement["seconds"]
            device_outputs = measurement["outputs"]
            correctness_checks = 10 if case.startswith("cutlass_gemm") else 6
            times = [wall]
        observed = [json.loads(line) for line in Path(str(commands) + ".cxx").read_text().splitlines()]
        hit_keys = re.findall(r"\[kernelbot-pch\] hit (\w+)", log)
        misses = "[kernelbot-pch] miss" in log
        expected_hit = enabled and case in CACHED_CASES
        assert bool(hit_keys) == expected_hit, log[-14000:]
        assert misses == (enabled and case == "flag_miss"), log[-14000:]
        # Independently verify actual GCC consumption after all timing/correctness checks.
        # This extra compilation and its header tracing are excluded from measured times.
        args = observed[0]["args"]
        args[args.index("-o") + 1] = str(root / "diagnostic.o")
        diagnostic = subprocess.run(
            [os.environ["PCH_TEST_COMPILER"], *args, "-H"], capture_output=True, text=True,
        )
        assert diagnostic.returncode == 0, diagnostic.stderr[-10000:]
        consumed = "! /kernelbot-pch/" in diagnostic.stderr
        assert consumed == expected_hit, diagnostic.stderr[-10000:]
        cuda_commands = Path(str(commands) + ".nvcc")
        cuda_times = [json.loads(line)["seconds"] for line in cuda_commands.read_text().splitlines()] \
            if cuda_commands.exists() else []
        return {
            "case": case, "variant": variant, "enabled": enabled, "repeat": repeat, "extension_name": name,
            "seconds": wall, "timing_samples": times,
            "host_compile_seconds": statistics.median(row["seconds"] for row in observed),
            "pch_consumed": consumed, "hit_keys": hit_keys, "cache_miss": misses,
            "correctness_checks": correctness_checks, "outputs": device_outputs,
            "source_sha256": hashlib.sha256((cpp + cuda + header).encode()).hexdigest(),
            "container_id": os.environ["MODAL_TASK_ID"], "image_id": os.environ["MODAL_IMAGE_ID"],
            "gpu": torch.cuda.get_device_name(), "torch": str(torch.__version__),
            "nvcc_seconds": cuda_times,
            "max_absolute_error": max(measurement["errors"], default=0) if case.startswith("cutlass_gemm") else 0,
            "remote_seconds": time.perf_counter() - started,
        }


@app.local_entrypoint()
def main(output: str = "/tmp/kernelbot-pch-matrix.json", cases: str = "", repeats: int = 1):
    selected = cases.split(",") if cases else CASES
    if not set(selected) <= set(CASES):
        raise ValueError(f"Choose from {CASES}")
    print(warm_pch.remote(profiles="cuda-default"))
    jobs = [
        (case, variant, enabled, repeat)
        for repeat in range(repeats)
        for case in selected
        for variant in (0, 1)
        for enabled in ((False, True) if variant == 0 else (True, False))
    ]
    rows = []
    for row in trial.starmap(jobs):
        rows.append(row)
        Path(output).write_text(json.dumps(rows, indent=2) + "\n")
        print(json.dumps(row), flush=True)
    assert len({row["container_id"] for row in rows}) == len(rows)
    hit_keys = {key for row in rows for key in row["hit_keys"]}
    assert len(hit_keys) == (1 if set(selected) & CACHED_CASES else 0)
    for case in selected:
        group = [row for row in rows if row["case"] == case]
        assert len({row["source_sha256"] for row in group}) == 2
        assert group[0]["outputs"] != group[2]["outputs"], "Expected distinct results from changed submissions"
        cold = statistics.median(row["seconds"] for row in group if not row["enabled"])
        warm = statistics.median(row["seconds"] for row in group if row["enabled"])
        print(f"{case}: baseline={cold:.4f}s cache={warm:.4f}s ratio={cold / warm:.3f}x")
