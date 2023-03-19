import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt


def RK45(x, func, h):
    """
    Runge-Kutta 四级四街法
    :param x: x = [x, y, z]
    :param func: 常微分方程右端的函数f
    :param h: 数值迭代的步长
    :return: 下一个点 x1 = [x, y, z]
    """
    K1 = func(x)
    K2 = func(x + h / 2 * K1)
    K3 = func(x + h / 2 * K2)
    K4 = func(x + h * K3)
    x1 = x + h / 6 * (K1 + 2 * K2 + 2 * K3 + K4)
    return x1


def L63_rhs(x):
    """
    ODE右端项 : f(t, u) ，lorenz右端没有t， 所以是一元函数 f(u)
    :param x: 当前所在点，[x, y, z]
    :return: 计算右端项的值， 对于lorenz方程组，就相当于 [dx/dt, dy/dt ,dz/dt]
    """
    #
    dx = np.ones_like(x)
    sigma = 10.0
    rho = 28.0
    beta = 8 / 3  # default parameters
    dx[0] = sigma * (x[1] - x[0])
    dx[1] = rho * x[0] - x[1] - x[0] * x[2]
    dx[2] = x[0] * x[1] - beta * x[2]
    return dx


def L63_adv_1step(x0, delta_t):
    """
    使用RK45求解ODE，从初值x0求得步长delta_t后的x状态，
    确切的说，是给一个点，求下一个点。
    :param x0: 当前点
    :param delta_t: 步长
    :return: 下一个点
    """

    x1 = RK45(x0, L63_rhs, delta_t)
    return x1


# 模式积分
# x0 为初值, 三个分量为 x, y, z 的初值
total_points = 300
x0 = np.array([1.508870, -1.531271, 25.46091])
# 计算三千个个点
Xtrue = np.zeros([total_points, 3])
Xtrue[0] = x0
# 步长
delta_t = 0.01
for j in range(1, total_points):
    Xtrue[j] = L63_adv_1step(Xtrue[j - 1], delta_t)

# 画图
mpl.rcParams['legend.fontsize'] = 10
fig = plt.figure()
# ax = fig.gca(projection='3d')
ax = fig.add_subplot(projection='3d')
ax.plot(Xtrue[:, 0], Xtrue[:, 1], Xtrue[:, 2], 'r', label='Lorenz 63 model')
ax.legend()
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.show()

# 模式积分
x0p = x0 + 0.001
Xctl = np.zeros([total_points, 3])
Xctl[0] = x0p
for j in range(1, total_points):
    Xctl[j] = L63_adv_1step(Xctl[j - 1], delta_t)
# 画图部分
fig = plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(Xtrue[range(total_points // 3), 1], Xtrue[range(total_points // 3), 2], 'r', label='Truth')
plt.plot(Xtrue[0, 1], Xtrue[0, 2], 'bx', ms=10, mew=3)
plt.plot(Xtrue[total_points // 3, 1], Xtrue[total_points // 3, 2], 'bo', ms=10)
plt.ylim(0, 50)
plt.title('True', fontsize=15)
plt.ylabel('z')
plt.xlabel('y')
plt.text(5, 25, r'$x_0$', fontsize=14)
plt.grid()
plt.subplot(1, 2, 2)
plt.plot(Xctl[range(total_points // 3), 1], Xctl[range(total_points // 3), 2], 'g')
plt.plot(Xctl[0, 1], Xctl[0, 2], 'bx', ms=10, mew=3, label='t0')
plt.plot(Xctl[total_points // 3, 1], Xctl[total_points // 3, 2], 'bo', ms=10, label='t')
plt.ylim(0, 50);
plt.title('Control', fontsize=15)
plt.ylabel('z')
plt.xlabel('y')
plt.grid()
plt.legend()
plt.text(5, 25, r'$x_0+0.001$', fontsize=14)
plt.show()
