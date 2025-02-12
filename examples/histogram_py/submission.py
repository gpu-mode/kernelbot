import torch
import triton
import triton.language as tl
from task import input_t, output_t, HistogramSpec

@triton.jit
def histogram_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    num_bins,
    min_val,
    max_val,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Parallel histogram kernel.
    Each thread block processes BLOCK_SIZE elements and maintains a local histogram,
    then atomically adds to the global histogram.
    """
    # Program ID
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load data
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    
    # Clip values to range
    x = tl.minimum(tl.maximum(x, min_val), max_val)
    
    # Convert to bin indices
    bin_width = (max_val - min_val) / num_bins
    indices = ((x - min_val) / bin_width).to(tl.int32)
    indices = tl.minimum(tl.maximum(indices, 0), num_bins - 1)
    
    # Initialize local histogram in shared memory
    local_hist = tl.zeros([num_bins], dtype=tl.float32)
    
    # Populate local histogram
    for i in range(BLOCK_SIZE):
        if offsets[i] < n_elements:
            bin_idx = indices[i]
            tl.atomic_add(local_hist + bin_idx, 1.0)
    
    # Add local histogram to global histogram
    for bin_idx in range(num_bins):
        if local_hist[bin_idx] > 0:
            tl.atomic_add(output_ptr + bin_idx, local_hist[bin_idx])

def custom_kernel(data: input_t, spec: HistogramSpec) -> output_t:
    """
    Computes histogram using parallel reduction.
    Args:
        data: Input tensor
        spec: Histogram specifications
    Returns:
        Tensor containing bin counts
    """
    n_elements = data.numel()
    
    # Initialize output histogram
    output = torch.zeros(spec.num_bins, device=data.device, dtype=torch.float32)
    
    # Configure kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Launch kernel
    histogram_kernel[grid](
        data,
        output,
        n_elements,
        spec.num_bins,
        spec.min_val,
        spec.max_val,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output 