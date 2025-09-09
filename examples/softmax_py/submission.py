#!POPCORN leaderboard softmax_py-dev

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    return torch.nn.functional.softmax(data, dim=-1)
