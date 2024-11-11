import torch



# Vector addition
a = torch.tensor([1, 2, 3]).cuda()
b = torch.tensor([4, 5, 6]).cuda()
c = a + b

print(a)
print(b)
print(c)
