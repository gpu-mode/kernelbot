import torch

x = torch.randn(2, 3)
y = torch.randn(3, 2)
z = torch.matmul(x, y)
print(z)