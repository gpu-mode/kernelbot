#!POPCORN leaderboard vectoradd_py-dev
"""FP16 vector add: keep Torch headers in C++, and only CUDA headers in CUDA.

Kernelbot's Modal CXX wrapper reuses the C++ torch header PCH automatically.
no_implicit_headers avoids unnecessarily parsing torch/types.h again with nvcc.
"""

from task import input_t, output_t
from torch.utils.cpp_extension import load_inline

add_cpp_source = r"""
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

void add_cuda_impl(const void*, const void*, void*, int, cudaStream_t);

torch::Tensor add_cuda(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && A.device() == B.device());
    TORCH_CHECK(A.scalar_type() == torch::kFloat16 && B.scalar_type() == torch::kFloat16);
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous() && A.sizes() == B.sizes());
    c10::cuda::CUDAGuard guard(A.device());
    auto C = torch::empty_like(A);
    add_cuda_impl(A.data_ptr(), B.data_ptr(), C.data_ptr(), A.numel(),
                  c10::cuda::getCurrentCUDAStream().stream());
    TORCH_CHECK(cudaGetLastError() == cudaSuccess, "CUDA launch failed");
    return C;
}
"""

add_cuda_source = r"""
#include <cuda_runtime.h>
#include <cuda_fp16.h>

__global__ void add_kernel(const __half* A, const __half* B, __half* C, int N) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) C[idx] = __hadd(A[idx], B[idx]);
}

void add_cuda_impl(const void* A, const void* B, void* C, int N, cudaStream_t stream) {
    const int threads = 256;
    add_kernel<<<(N + threads - 1) / threads, threads, 0, stream>>>(
        static_cast<const __half*>(A), static_cast<const __half*>(B), static_cast<__half*>(C), N);
}
"""

add_module = load_inline(
    name='add_cuda',
    cpp_sources=add_cpp_source,
    cuda_sources=add_cuda_source,
    functions=['add_cuda'],
    no_implicit_headers=True,
    verbose=True,
)


def add(A, B):
    return add_module.add_cuda(A, B)


def custom_kernel(data: input_t) -> output_t:
    return add(*data)
