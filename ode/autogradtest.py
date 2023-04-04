# my python template
# created by leay
# date: 2023/4/3 14:34

import torch
import torch.nn as nn
import numpy as np
import torch.autograd as autograd

# 区间总数
total_interval = 1500
# 总点数
total_points = total_interval + 1
# 区间长度
h = 0.005
x_lb = torch.tensor(0.)
x_ub = torch.tensor(total_interval * h)

x_test = torch.linspace(x_lb, x_ub, total_points)
x_test.requires_grad = True
y_test = torch.sin(x_test)

y_t = autograd.grad(y_test, x_test, torch.ones_like(y_test), create_graph=True)[0]

print(y_t[:20])
print(torch.cos(x_test)[:20])
