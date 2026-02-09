"""Test script to verify CUTLASS C++ headers work on a Modal image.

Usage:
    cd src/runners
    modal run test_cutlass_image.py

This builds a test image with CUTLASS v4.3.5 headers and runs a simple
compilation test on a GPU. Does NOT affect the production image.
"""

import modal

app = modal.App("test-cutlass-image")

cuda_version = "13.1.0"
flavor = "devel"
operating_sys = "ubuntu24.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

test_image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.13")
    .run_commands("ln -sf $(which python) /usr/local/bin/python3")
    .apt_install("git", "gcc-13", "g++-13")
    .uv_pip_install("ninja~=1.11")
    .uv_pip_install(
        "torch==2.9.1",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    # CUTLASS C++ headers
    .run_commands(
        "git clone --depth 1 --branch v4.3.5 https://github.com/NVIDIA/cutlass.git /opt/cutlass",
    )
    .env({
        "CUTLASS_PATH": "/opt/cutlass",
        "CPLUS_INCLUDE_PATH": "/opt/cutlass/include:/opt/cutlass/tools/util/include",
    })
)


CUTLASS_TEST_CU = r"""
#include <iostream>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

int main() {
    std::cout << "CUTLASS include works!" << std::endl;

    // Test CuTe tensor layout (core CUTLASS 3.x/4.x API)
    auto layout = cute::make_layout(cute::make_shape(4, 8));
    std::cout << "CuTe layout size: " << cute::size(layout) << std::endl;

    // Test CUTLASS numeric types
    cutlass::half_t h = cutlass::half_t(3.14f);
    std::cout << "half_t value: " << float(h) << std::endl;

    std::cout << "All CUTLASS tests passed!" << std::endl;
    return 0;
}
"""


@app.function(gpu="T4", image=test_image, timeout=300)
def test_cutlass():
    import subprocess
    import tempfile
    import os

    results = {}

    # Test 1: Check that CUTLASS headers exist
    cutlass_path = os.environ.get("CUTLASS_PATH", "")
    header = os.path.join(cutlass_path, "include", "cutlass", "cutlass.h")
    results["cutlass_path"] = cutlass_path
    results["header_exists"] = os.path.exists(header)

    # Test 2: Check CPLUS_INCLUDE_PATH is set
    results["cplus_include_path"] = os.environ.get("CPLUS_INCLUDE_PATH", "NOT SET")

    # Test 3: Compile and run a simple CUTLASS program
    with tempfile.TemporaryDirectory() as tmpdir:
        cu_file = os.path.join(tmpdir, "test_cutlass.cu")
        binary = os.path.join(tmpdir, "test_cutlass")

        with open(cu_file, "w") as f:
            f.write(CUTLASS_TEST_CU)

        compile_cmd = [
            "nvcc",
            cu_file,
            "-o", binary,
            "-I", f"{cutlass_path}/include",
            "-I", f"{cutlass_path}/tools/util/include",
            "-std=c++17",
            "-arch=sm_75",
        ]

        compile_result = subprocess.run(
            compile_cmd, capture_output=True, text=True
        )
        results["compile_returncode"] = compile_result.returncode
        results["compile_stdout"] = compile_result.stdout
        results["compile_stderr"] = compile_result.stderr

        if compile_result.returncode == 0:
            run_result = subprocess.run(
                [binary], capture_output=True, text=True
            )
            results["run_returncode"] = run_result.returncode
            results["run_stdout"] = run_result.stdout
            results["run_stderr"] = run_result.stderr

    # Test 4: Check that PyTorch CUDA extension loading works with CUTLASS
    torch_cutlass_test = """
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = '''
#include <cutlass/cutlass.h>
#include <torch/extension.h>

torch::Tensor check_cutlass(torch::Tensor x) {
    // Just return input - proves cutlass headers are findable
    return x;
}
'''

cpp_src = "torch::Tensor check_cutlass(torch::Tensor x);"

try:
    mod = load_inline(
        name="cutlass_check",
        cpp_sources=[cpp_src],
        cuda_sources=[cuda_src],
        extra_include_paths=["/opt/cutlass/include", "/opt/cutlass/tools/util/include"],
        functions=["check_cutlass"],
        verbose=True,
    )
    t = torch.randn(4, device="cuda")
    result = mod.check_cutlass(t)
    print(f"SUCCESS: PyTorch inline CUDA extension with CUTLASS compiled and ran. Output shape: {result.shape}")
except Exception as e:
    print(f"FAILED: {e}")
"""
    torch_result = subprocess.run(
        ["python", "-c", torch_cutlass_test],
        capture_output=True, text=True
    )
    results["torch_extension_returncode"] = torch_result.returncode
    results["torch_extension_stdout"] = torch_result.stdout
    results["torch_extension_stderr"] = torch_result.stderr[-2000:] if len(torch_result.stderr) > 2000 else torch_result.stderr

    return results


@app.local_entrypoint()
def main():
    print("=" * 60)
    print("Testing CUTLASS C++ headers on Modal (T4 GPU)")
    print("=" * 60)

    results = test_cutlass.remote()

    print(f"\n--- Environment ---")
    print(f"CUTLASS_PATH: {results['cutlass_path']}")
    print(f"Header exists: {results['header_exists']}")
    print(f"CPLUS_INCLUDE_PATH: {results['cplus_include_path']}")

    print(f"\n--- nvcc Compilation Test ---")
    print(f"Return code: {results['compile_returncode']}")
    if results['compile_stdout']:
        print(f"stdout: {results['compile_stdout']}")
    if results['compile_stderr']:
        print(f"stderr: {results['compile_stderr']}")

    if results.get('run_returncode') is not None:
        print(f"\n--- Run Test ---")
        print(f"Return code: {results['run_returncode']}")
        print(f"stdout: {results['run_stdout']}")
        if results['run_stderr']:
            print(f"stderr: {results['run_stderr']}")

    print(f"\n--- PyTorch Inline Extension Test ---")
    print(f"Return code: {results['torch_extension_returncode']}")
    if results['torch_extension_stdout']:
        print(f"stdout: {results['torch_extension_stdout']}")
    if results['torch_extension_stderr']:
        print(f"stderr (last 2000 chars): {results['torch_extension_stderr']}")

    # Summary
    print("\n" + "=" * 60)
    all_pass = (
        results['header_exists']
        and results['compile_returncode'] == 0
        and results.get('run_returncode') == 0
        and results['torch_extension_returncode'] == 0
    )
    if all_pass:
        print("ALL TESTS PASSED - safe to add CUTLASS to production image")
    else:
        print("SOME TESTS FAILED - check output above")
    print("=" * 60)
