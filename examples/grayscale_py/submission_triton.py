import torch
import triton
import triton.language as tl
from task import input_t, output_t

@triton.jit
def grayscale_kernel(
    input_ptr, output_ptr,
    H, W,
    stride_h, stride_w, stride_c,
    BLOCK_SIZE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate start indices
    block_start_h = (pid // ((W + BLOCK_SIZE - 1) // BLOCK_SIZE)) * BLOCK_SIZE
    block_start_w = (pid % ((W + BLOCK_SIZE - 1) // BLOCK_SIZE)) * BLOCK_SIZE
    
    # Offsets for this block
    offs_h = block_start_h + tl.arange(0, BLOCK_SIZE)
    offs_w = block_start_w + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid pixels
    mask = (offs_h[:, None] < H) & (offs_w[None, :] < W)
    
    # RGB to Grayscale coefficients
    R_COEF = 0.2989
    G_COEF = 0.5870
    B_COEF = 0.1140
    
    # Calculate base pointer for each pixel in the block
    base_ptr = offs_h[:, None] * stride_h + offs_w[None, :] * stride_w
    
    # Load RGB channels
    r = tl.load(input_ptr + base_ptr + 0 * stride_c, mask=mask, other=0.0)
    g = tl.load(input_ptr + base_ptr + 1 * stride_c, mask=mask, other=0.0)
    b = tl.load(input_ptr + base_ptr + 2 * stride_c, mask=mask, other=0.0)
    
    # Convert to grayscale
    gray = R_COEF * r + G_COEF * g + B_COEF * b
    
    # Store result
    out_ptr = offs_h[:, None] * W + offs_w[None, :]
    tl.store(output_ptr + out_ptr, gray, mask=mask)

def custom_kernel(data: input_t) -> output_t:
    H, W, C = data.shape
    assert C == 3, "Input must be an RGB image"
    
    # Create output tensor
    output = torch.empty((H, W), device=data.device, dtype=data.dtype)
    
    # Calculate strides
    stride_h = W * C
    stride_w = C
    stride_c = 1
    
    # Launch kernel
    BLOCK_SIZE = 32
    grid = ((H + BLOCK_SIZE - 1) // BLOCK_SIZE) * ((W + BLOCK_SIZE - 1) // BLOCK_SIZE)
    
    grayscale_kernel[grid](
        data, output,
        H, W,
        stride_h, stride_w, stride_c,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output 