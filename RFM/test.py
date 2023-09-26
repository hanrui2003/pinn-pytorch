import torch
import numpy as np

# 创建一个 NumPy 数组
numpy_array = np.array([1.0, 2.0, 3.0])

# 使用 torch.tensor 从 NumPy 数组创建一个张量
point = torch.tensor(numpy_array)

# 修改 point 中的值，不会影响原始的 numpy_array
point[0] = 10.0
print(point)  # 输出 tensor([10.,  2.,  3.])
print(numpy_array)  # 输出 [ 1.  2.  3.]
