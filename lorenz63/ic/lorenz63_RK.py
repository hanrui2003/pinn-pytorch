import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt


def RK45(u, f, h):
    """
    Runge-Kutta 四级四街法
    :param u: [x, y, z]
    :param f: 常微分方程右端的函数f
    :param h: 数值迭代的步长
    :return: 下一个点 [x, y, z]
    """
    k1 = f(u)
    k2 = f(u + h / 2 * k1)
    k3 = f(u + h / 2 * k2)
    k4 = f(u + h * k3)
    return u + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def lorenz_f(u):
    """
    ODE右端项 : f(t, u) ，lorenz右端没有t， 所以是一元函数 f(u)
    :param u: 当前所在点，[x, y, z]
    :return: 计算ODE右端项的值， 对于lorenz方程组，就相当于 [dx/dt, dy/dt ,dz/dt]
    """
    dt = np.empty_like(u)
    # default parameters

    dt[0] = sigma * (u[1] - u[0])
    dt[1] = rho * u[0] - u[1] - u[0] * u[2]
    dt[2] = u[0] * u[1] - beta * u[2]
    return dt


def u_next(u, h):
    """
    使用RK45求解ODE， 确切的说，是给一个点，求下一个点。
    :param u: 当前点
    :param h: 步长
    :return: 下一个点
    """
    return RK45(u, lorenz_f, h)


sigma = 10.0
# 最后趋于稳态的
rho = 15.0
# 一直震荡的
# rho = 28.0
beta = 8 / 3

# 总点数
total_points = 601
# 步长
h = 0.005

t = np.arange(0., total_points * h, h)
u = np.zeros([total_points, 3])
# 初值
u[0] = [-4., 7., 15]

for j in range(1, total_points):
    u[j] = u_next(u[j - 1], h)

du = []
for p in u:
    du.append((sigma * (p[1] - p[0])) ** 2 + (rho * p[0] - p[1] - p[0] * p[2]) ** 2 + (
            p[0] * p[1] - beta * p[2]) ** 2)

print(du)
print(min(du))
# 画图
fig = plt.figure()
ax = fig.add_subplot()

ax.plot(t, u[:, 0], color='r', label='x')
ax.plot(t, u[:, 1], color='g', label='y')
ax.plot(t, u[:, 2], color='b', label='z')
ax.legend(loc='upper right')
plt.savefig('./images/lorenz63_RK_2d.png')
plt.show()

ax = plt.axes(projection='3d')
ax.plot(u[:, 0], u[:, 1], u[:, 2], color='r')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.set_title('lorenz63')
plt.savefig('./images/lorenz63_RK_3d.png')
plt.show()
