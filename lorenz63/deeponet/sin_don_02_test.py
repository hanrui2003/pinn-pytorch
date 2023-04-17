import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

from sin_don_01 import PIDeepONet

PIDON = torch.load('sin_don_02.pt')
print(PIDON)

# 测试点
n_t = 101
t_test = torch.linspace(0, 1, n_t)

y_predict_total = torch.Tensor()
for theta in range(10):
    theta_test = torch.tensor(theta, dtype=torch.float).tile(n_t, 1)
    y_predict = PIDON(t_test.unsqueeze(1), theta_test).detach()
    y_predict_total = torch.cat((y_predict_total, y_predict[:-1]), dim=0)

t_test_total = torch.linspace(0, 9.99, 1000)
fig, ax = plt.subplots()
ax.plot(t_test_total, y_predict_total, 'r-', linewidth=2, label='predict')
ax.plot(t_test_total, np.sin(t_test_total), 'b--', linewidth=2, label='x-truth')
ax.set_title('sin t')
ax.set_xlabel('t', color='black')
ax.set_ylabel('f(t)', color='black', rotation=0)
ax.legend(loc='upper right')
plt.show()
