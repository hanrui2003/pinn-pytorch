import torch
import numpy as np
import matplotlib.pyplot as plt
from simple_ode_02 import FCN

layers = [1, 32, 32, 2]
PINN = FCN(layers)
PINN.load_state_dict(torch.load('simple_ode_02.pt'))
print("xzcvzxcvasdf")
print(PINN)

t_lb = 0.0
t_ub = 2 * np.pi
total_points = 500

x_test = torch.linspace(t_lb, t_ub, total_points)

nn_predict = PINN(x_test[:, None]).detach().numpy()
x_predict = nn_predict[:, 0]
y_predict = nn_predict[:, 1]

fig, ax = plt.subplots()

ax.plot(x_test, x_predict, color='r', label='f1')
ax.plot(x_test, y_predict, color='g', label='f2')
ax.set_xlabel('x', color='black')
ax.set_ylabel('f(x)', color='black')
ax.legend(loc='upper left')
plt.savefig('./figure/ODEs_02_load_2d.png')
plt.show()
