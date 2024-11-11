import torch

a = torch.Tensor([1, 2, 3, 4, 5]).to('cuda')
b= torch.Tensor([1, 2, 3, 4, 5]).to('cuda')
print(a + b)