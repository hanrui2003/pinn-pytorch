import torch

# 创建两个二维张量
a = torch.tensor([[1, 2], [3, 4], [5, 6]])
b = torch.tensor([[7, 8], [9, 10], [11, 12]])

# 计算对应行向量的内积
c = torch.sum(torch.mul(a, b), dim=1)

# 输出结果
print(c)

