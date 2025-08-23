import torch
from task import input_t, output_t
from utils import verbose_allclose
from typing import Tuple


def generate_input(seed: int, world_size: int, rank: int) -> input_t:
    local_data = torch.tensor([rank]).to(f"cuda:{rank}")
    return local_data, rank, world_size


def check_implementation(data: input_t, output: output_t) -> Tuple[bool, str]:
    data, rank, world_size = data
    for i in range(world_size):
        if output[i].get_device() != rank:
            return False, f"mismatch found! output {i} of rank {rank} is on device {output[i].device}"
        if (item := output[i].cpu().detach().item()) != i:
            return False, f"mismatch found! custom implementation doesn't match reference: rank {rank}, entry {i} has value {item}"
    return True, ''
