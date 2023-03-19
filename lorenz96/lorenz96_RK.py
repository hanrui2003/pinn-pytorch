import numpy as np
import matplotlib.pyplot as plt

# 方程个数
N = 40
# 扰动项
F = 8


def RK45(f, x, h):
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    return x + (k1 + 2 * k2 + 2 * k3 + k4) * h / 6


def f(x):
    dx = np.empty(N)
    for i in range(N):
        dx[i] = (x[(i + 1) % N] - x[i - 2]) * x[i - 1] - x[i] + F
    return dx


# 点数
total_point = 1000
# 步长
h = 0.01
t = np.arange(0.0, total_point * h, h)
x = np.empty((total_point, N))
x0 = np.zeros(N) + F
x0[0] += 0.01
x[0] = x0

for j in range(1, total_point):
    x[j] = RK45(f, x[j - 1], h)

fig, ax = plt.subplots()

ax.plot(t, x[:, 0], color='r', label='x')
ax.plot(t, x[:, 1], color='g', label='y')
ax.plot(t, x[:, 2], color='b', label='z')
ax.set_title('Lorenz 96')
ax.set_xlabel('t', color='black')
ax.set_ylabel('f(t)', color='black', rotation=0)
ax.legend(loc='upper right')
plt.savefig('./figure/lorenz96_RK_2d.png')
# plt.close()
plt.show()

ax = plt.axes(projection="3d")
ax.plot(x[:, 0], x[:, 1], x[:, 2])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
ax.set_zlabel('$x_3$')
plt.savefig('./figure/lorenz96_RK_3d.png')
plt.show()

# fig, ax = plt.subplots()
# ax.plot(x_test, x_nn, color='r', label='x')
# ax.plot(x_test, y_nn, color='g', label='y')
# ax.plot(x_test, z_nn, color='b', label='z')
# ax.set_title('Lorenz with 300 point')
# ax.set_xlabel('t', color='black')
# ax.set_ylabel('f(t)', color='black', rotation=0)
# ax.legend(loc='upper right')
# plt.savefig('./figure/lorenz63_pinn_06_2d.png')
# plt.close()
# # plt.show()
