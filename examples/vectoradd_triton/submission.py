import torch
import triton
import triton.language as tl
from typing import List
from task import kernel_interface

@triton.jit
def add_kernel(
    a_ptr, b_ptr, c_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    
    c = a + b
    
    tl.store(c_ptr + offsets, c, mask=mask)

def custom_kernel(inputs: List[List[torch.Tensor]]) -> List[torch.Tensor]:
    """
    Custom implementation of vector addition using Triton.
    Args:
        inputs: List of pairs of tensors [A, B] to be added.
    Returns:
        List of tensors containing element-wise sums.
    """
    results = []
    for A, B in inputs:
        assert A.is_cuda and B.is_cuda, "Input tensors must be on GPU"
        assert A.shape == B.shape, "Input tensors must have the same shape"
        assert A.dtype == torch.float16 and B.dtype == torch.float16, "Input tensors must be float16"
        
        M, N = A.shape
        n_elements = M * N
        C = torch.empty_like(A)
        
        def grid(meta): return (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        
        add_kernel[grid](
            A.reshape(-1).data_ptr(),
            B.reshape(-1).data_ptr(),
            C.reshape(-1).data_ptr(),
            n_elements,
            BLOCK_SIZE=1024
        )
        
        results.append(C)
    
    return results
