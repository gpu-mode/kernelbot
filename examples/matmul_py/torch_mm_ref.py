#!POPCORN leaderboard matmul_py

from task import input_t, output_t
import torch


def custom_kernel(data: input_t) -> output_t:
    a, b = data
    return torch.mm(a, b)
