from lorenz63_dnn_obs_03 import FCN
import torch
import matplotlib.pyplot as plt

PINN = torch.load('lorenz63_dnn_obs_03.pt')
print(PINN)

# 区间总数
total_interval = 600
# 总点数
total_points = total_interval + 1
# 区间长度
h = 0.005
x_lb = torch.tensor(0.)
x_ub = torch.tensor(total_interval * h)

# lorenz 方程参数
rho = torch.tensor(15.0)
sigma = torch.tensor(10.0)
beta = torch.tensor(8.0 / 3.0)

x_test = torch.linspace(x_lb, x_ub, total_points)

nn_predict = PINN(x_test[:, None]).detach().numpy()
x_nn = nn_predict[:, [0]]
y_nn = nn_predict[:, [1]]
z_nn = nn_predict[:, [2]]

fig, ax = plt.subplots()
ax.plot(x_test, x_nn, color='r', label='x')
ax.plot(x_test, y_nn, color='g', label='y')
ax.plot(x_test, z_nn, color='b', label='z')
ax.set_title('Lorenz63')
ax.set_xlabel('t', color='black')
ax.set_ylabel('f(t)', color='black', rotation=0)
ax.legend(loc='upper right')
plt.show()

ax = plt.axes(projection="3d")
ax.plot(x_nn, y_nn, z_nn, color='r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('lorenz63')
plt.show()
