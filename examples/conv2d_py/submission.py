import torch
import triton
import triton.language as tl
from task import input_t, output_t, KernelSpec

@triton.jit
def conv2d_kernel(
    # Pointers to matrices
    input_ptr, kernel_ptr, output_ptr,
    # Matrix dimensions
    batch, in_channels, out_channels, 
    in_height, in_width,
    kernel_size, stride, padding,
    out_height, out_width,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """
    2D Convolution kernel.
    Each thread block handles computation for a BLOCK_SIZE_M x BLOCK_SIZE_N region of the output.
    """
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate output position
    n_blocks_m = triton.cdiv(out_height, BLOCK_SIZE_M)
    batch_idx = pid // (n_blocks_m * out_channels)
    tmp = pid % (n_blocks_m * out_channels)
    out_ch = tmp // n_blocks_m
    block_m = tmp % n_blocks_m
    
    # Calculate output row and column ranges for this block
    out_m = block_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    out_n = tl.arange(0, BLOCK_SIZE_N)
    
    # Calculate input positions with padding offset
    in_m = out_m * stride - padding
    in_n = out_n * stride - padding
    
    # Initialize output accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate over input channels and kernel positions
    for in_ch in range(in_channels):
        for kh in range(kernel_size):
            for kw in range(kernel_size):
                # Calculate input positions
                h_pos = in_m + kh
                w_pos = in_n + kw
                
                # Create masks for valid positions
                m_mask = (h_pos >= 0) & (h_pos < in_height)
                n_mask = (w_pos >= 0) & (w_pos < in_width)
                mask = m_mask[:, None] & n_mask[None, :]
                
                # Load input values
                x_pos = h_pos[:, None] * in_width + w_pos[None, :]
                input_idx = ((batch_idx * in_channels + in_ch) * in_height * in_width + x_pos)
                x = tl.load(input_ptr + input_idx, mask=mask, other=0.0)
                
                # Load kernel value
                k_idx = ((out_ch * in_channels + in_ch) * kernel_size * kernel_size + 
                        kh * kernel_size + kw)
                k = tl.load(kernel_ptr + k_idx)
                
                # Accumulate
                acc += k * x
    
    # Write output
    out_pos = out_m[:, None] * out_width + out_n[None, :]
    output_idx = ((batch_idx * out_channels + out_ch) * out_height * out_width + 
                  out_pos)
    
    # Create output mask
    m_mask = out_m < out_height
    n_mask = out_n < out_width
    mask = m_mask[:, None] & n_mask[None, :]
    
    # Store output
    tl.store(output_ptr + output_idx, acc, mask=mask)

def custom_kernel(data: input_t, spec: KernelSpec) -> output_t:
    """
    Performs 2D convolution using Triton kernel.
    Args:
        data: Tuple of (input tensor, kernel tensor)
        spec: Convolution specifications
    Returns:
        Output tensor after convolution
    """
    input_tensor, kernel = data
    batch, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_size, _ = kernel.shape
    
    # Calculate output dimensions
    out_height = ((in_height + 2 * spec.padding - kernel_size) // spec.stride) + 1
    out_width = ((in_width + 2 * spec.padding - kernel_size) // spec.stride) + 1
    
    # Allocate output
    output = torch.empty(
        (batch, out_channels, out_height, out_width),
        device=input_tensor.device,
        dtype=input_tensor.dtype
    )
    
    # Configure kernel
    BLOCK_SIZE_M = 8
    BLOCK_SIZE_N = 8
    grid = (batch * out_channels * triton.cdiv(out_height, BLOCK_SIZE_M),)
    
    # Launch kernel
    conv2d_kernel[grid](
        input_tensor, kernel, output,
        batch, in_channels, out_channels,
        in_height, in_width,
        kernel_size, spec.stride, spec.padding,
        out_height, out_width,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    return output 