import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def scan_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Single-block inclusive prefix sum kernel.
    Uses a two-pass approach: up-sweep and down-sweep.
    """
    # Get program ID and allocate shared memory
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data into shared memory
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Up-sweep: Build sum tree
    offset = 1
    for d in range(triton.next_power_of_2(BLOCK_SIZE) // 2):
        mask = tl.arange(0, BLOCK_SIZE) % (2 * offset) == (2 * offset - 1)
        vals = tl.where(mask, x, 0.0)
        vals = tl.sum(vals, axis=0)
        x = x + tl.where(mask, -x + vals, 0.0)
        offset *= 2
    
    # Down-sweep: Distribute sums
    for d in range(triton.next_power_of_2(BLOCK_SIZE) // 2 - 1, -1, -1):
        offset = 1 << d
        mask = tl.arange(0, BLOCK_SIZE) % (2 * offset) == (2 * offset - 1)
        vals = tl.where(mask, x, 0.0)
        x = x + tl.where(tl.arange(0, BLOCK_SIZE) % (2 * offset) >= offset, vals, 0.0)
    
    # Store results
    output_mask = offsets < n_elements
    tl.store(output_ptr + offsets, x, mask=output_mask)

@triton.jit
def block_sum_kernel(
    block_sums_ptr,
    output_ptr,
    block_size,
    n_blocks,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Adds block sums to subsequent blocks to get final prefix sum.
    """
    pid = tl.program_id(0)
    block_idx = pid + 1  # Skip first block
    
    if block_idx < n_blocks:
        # Load block sum from previous block
        prev_sum = tl.load(block_sums_ptr + block_idx - 1)
        
        # Add to all elements in current block
        offsets = block_idx * block_size + tl.arange(0, BLOCK_SIZE)
        mask = offsets < (block_idx + 1) * block_size
        x = tl.load(output_ptr + offsets, mask=mask, other=0.0)
        x = x + prev_sum
        tl.store(output_ptr + offsets, x, mask=mask)

def custom_kernel(data: input_t) -> output_t:
    """
    Multi-block prefix sum implementation.
    Args:
        data: Input tensor
    Returns:
        Tensor containing inclusive prefix sum
    """
    n_elements = data.numel()
    output = torch.empty_like(data)
    
    # Configure kernel
    BLOCK_SIZE = 1024
    n_blocks = triton.cdiv(n_elements, BLOCK_SIZE)
    
    # Phase 1: Compute prefix sum within each block
    scan_kernel[(n_blocks,)](
        data,
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    if n_blocks > 1:
        # Get block sums
        block_sums = torch.empty(n_blocks, device=data.device, dtype=data.dtype)
        block_sums[0] = output[BLOCK_SIZE-1]
        for i in range(1, n_blocks-1):
            block_sums[i] = output[(i+1)*BLOCK_SIZE-1]
        
        # Compute prefix sum of block sums
        block_sums = torch.cumsum(block_sums, dim=0)
        
        # Phase 2: Add block sums to subsequent blocks
        block_sum_kernel[(n_blocks-1,)](
            block_sums,
            output,
            BLOCK_SIZE,
            n_blocks,
            BLOCK_SIZE=BLOCK_SIZE,
        )
    
    return output 