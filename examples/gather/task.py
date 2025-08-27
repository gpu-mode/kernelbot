from typing import TypedDict, List, Tuple
import torch


input_t = Tuple[torch.Tensor, int, int]
output_t = List[torch.Tensor]


class TestSpec(TypedDict):
    pass
