import torch
import random
import numpy as np
from typing import List

def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def check_implementation(
    custom_outputs: List[torch.Tensor],
    reference_outputs: List[torch.Tensor],
    rtol: float = 1e-4,
    atol: float = 1e-4
) -> bool:
    """Check if custom implementation matches reference implementation."""
    if len(custom_outputs) != len(reference_outputs):
        return False
    
    for i, (custom_output, reference_output) in enumerate(zip(custom_outputs, reference_outputs)):
        if custom_output.shape != reference_output.shape:
            return False
        
        if not torch.allclose(custom_output, reference_output, rtol=rtol, atol=atol):
            max_diff = torch.max(torch.abs(custom_output - reference_output)).item()
            print(f"Maximum difference at output {i}: {max_diff}")
            return False
    
    return True 
