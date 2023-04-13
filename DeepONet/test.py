import torch

x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x ** 2
z = torch.sum(y)

grad_x = torch.autograd.grad(z, x,torch.ones_like(z), retain_graph=True)
print(grad_x)
