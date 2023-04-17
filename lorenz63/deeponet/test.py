import torch

# 创建一个空的张量
new_tensor = torch.Tensor()

# 循环追加多个张量
for i in range(5):
    tensor = torch.randn((3, 4))
    new_tensor = torch.cat((new_tensor, tensor), dim=0)

print(new_tensor)
