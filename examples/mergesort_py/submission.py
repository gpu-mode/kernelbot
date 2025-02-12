import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def merge_kernel(
    x_ptr,
    temp_ptr,
    n_elements,
    chunk_size,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Merges sorted chunks of size chunk_size into sorted chunks of size 2*chunk_size.
    Each thread block handles merging two adjacent sorted chunks.
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate start of the two chunks to merge
    chunk_pair_start = pid * (2 * chunk_size)
    if chunk_pair_start >= n_elements:
        return
        
    # Calculate sizes of chunks to merge (handle last chunk specially)
    left_size = min(chunk_size, n_elements - chunk_pair_start)
    right_start = chunk_pair_start + chunk_size
    right_size = min(chunk_size, n_elements - right_start) if right_start < n_elements else 0
    
    # Load left chunk
    left_idx = chunk_pair_start + tl.arange(0, BLOCK_SIZE)
    left_mask = left_idx < (chunk_pair_start + left_size)
    left = tl.load(x_ptr + left_idx, mask=left_mask, other=float('inf'))
    
    # Load right chunk
    right_idx = right_start + tl.arange(0, BLOCK_SIZE)
    right_mask = right_idx < (right_start + right_size)
    right = tl.load(x_ptr + right_idx, mask=right_mask, other=float('inf'))
    
    # Merge chunks using parallel merge path
    output = tl.zeros([2 * BLOCK_SIZE], dtype=tl.float32) + float('inf')
    left_ptr = 0
    right_ptr = 0
    out_ptr = 0
    
    for i in range(left_size + right_size):
        # Compare current elements
        take_left = (left_ptr < left_size and 
                    (right_ptr >= right_size or left[left_ptr] <= right[right_ptr]))
        
        # Store smaller element
        output[out_ptr] = tl.where(take_left, left[left_ptr], right[right_ptr])
        
        # Advance pointers
        left_ptr = left_ptr + tl.where(take_left, 1, 0)
        right_ptr = right_ptr + tl.where(take_left, 0, 1)
        out_ptr = out_ptr + 1
    
    # Store merged result
    out_idx = chunk_pair_start + tl.arange(0, 2 * BLOCK_SIZE)
    out_mask = out_idx < min(chunk_pair_start + left_size + right_size, n_elements)
    tl.store(temp_ptr + out_idx, output, mask=out_mask)

def custom_kernel(data: input_t) -> output_t:
    """
    Implements parallel merge sort.
    Args:
        data: Input tensor to be sorted
    Returns:
        Sorted tensor
    """
    n_elements = data.numel()
    if n_elements <= 1:
        return data.clone()
    
    # Allocate temporary buffer for merging
    temp = torch.empty_like(data)
    output = data.clone()
    
    # Configure kernel
    BLOCK_SIZE = 512  # Should be power of 2 for simplicity
    
    # Bottom-up merge sort
    chunk_size = BLOCK_SIZE
    while chunk_size < n_elements:
        n_chunk_pairs = triton.cdiv(n_elements, 2 * chunk_size)
        
        # Launch merge kernel
        merge_kernel[(n_chunk_pairs,)](
            output,
            temp,
            n_elements,
            chunk_size,
            BLOCK_SIZE=BLOCK_SIZE,
        )
        
        # Swap buffers
        output, temp = temp, output
        chunk_size *= 2
    
    return output 