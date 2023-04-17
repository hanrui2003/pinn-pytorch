import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt

from sin_don_01 import PIDeepONet

PIDON = torch.load('sin_don_01.pt')
print(PIDON)

initial = torch.tensor(0.3)

# 测试点
n_t = 600
t_test = torch.linspace(0, 2 * torch.pi, n_t)

initial_test = initial.tile(n_t, 1)

y_predict = PIDON(t_test.unsqueeze(1), initial_test).detach().numpy()
print(y_predict)

fig, ax = plt.subplots()
ax.plot(t_test, y_predict, 'r-', linewidth=2, label='predict')
ax.plot(t_test, np.sin(t_test) + initial, 'b--', linewidth=2, label='x-truth')
ax.set_title('sin t')
ax.set_xlabel('t', color='black')
ax.set_ylabel('f(t)', color='black', rotation=0)
ax.legend(loc='upper right')
plt.show()
