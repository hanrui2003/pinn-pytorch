import numpy as np
import matplotlib.pyplot as plt
import math


def lorenz63_rhs(u):
    """
    ODE右端项 : f(t, u) ，lorenz右端没有t， 所以是一元函数 f(u)
    :param u: 当前所在点，[x, y, z]
    :return: 计算右端项的值， 对于lorenz方程组，就相当于 [dx/dt, dy/dt ,dz/dt]
    """
    #
    du = np.ones_like(u)
    sigma = 10.0
    rho = 28.0
    beta = 8 / 3
    du[0] = sigma * (u[1] - u[0])
    du[1] = rho * u[0] - u[1] - u[0] * u[2]
    du[2] = u[0] * u[1] - beta * u[2]
    return du


if "__main__" == __name__:
    # 分数阶
    alpha = 0.99
    # 步长
    h = 0.005
    # 总点数
    total_points = 301
    t = np.arange(0., total_points * h, h)
    # u存储离散值
    u = np.zeros([total_points, 3])
    # 初值，对应蝴蝶状
    # u[0] = np.array([-4., 7., 15.])
    u[0] = np.array([0.55460613, -0.66932589, 20.8581486])
    # 隐式迭代收敛判断
    epsilon = 0.001
    for n in range(1, total_points):
        # 隐式迭代
        u_n_prev = u[n - 1].copy()
        while True:
            u[n] = math.gamma(2 - alpha) * h ** alpha * lorenz63_rhs(u_n_prev) + (
                    n ** (1 - alpha) - (n - 1) ** (1 - alpha)) * u[0]
            for k in range(1, n):
                u[n] -= ((n - k + 1) ** (1 - alpha) - 2 * (n - k) ** (1 - alpha) + (n - k - 1) ** (1 - alpha)) * u[k]

            if any(abs(u[n] - u_n_prev) > epsilon):
                u_n_prev = u[n].copy()
            else:
                break

    # 存储，用于神经网络的测试
    np.save(r'florenz63_rho_28_interval_0.005_total_point_301.npy', u)
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(t, u[:, 0], color='r', label='x')
    ax.plot(t, u[:, 1], color='g', label='y')
    ax.plot(t, u[:, 2], color='b', label='z')
    ax.legend(loc='upper right')
    plt.savefig('./figure/frac_ode_fdm_01_2d.png')
    plt.show()

    ax = plt.axes(projection='3d')
    ax.plot(u[:, 0], u[:, 1], u[:, 2], color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('florenz63')
    plt.savefig('./figure/frac_ode_fdm_01_3d.png')
    plt.show()
