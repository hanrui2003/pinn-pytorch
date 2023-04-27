import torch

# 定义函数
def my_func(x, y, z):
    output = x * y + z
    return output

# 创建输入张量
x = torch.tensor([1.0], requires_grad=True)
y = torch.tensor([2.0], requires_grad=True)
z = torch.tensor([3.0], requires_grad=True)

# 计算x关于output的梯度
grad_x = torch.autograd.grad(my_func(x, y, z), x, grad_outputs=torch.tensor([1.0]))[0]
print(grad_x)
