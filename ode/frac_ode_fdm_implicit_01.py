import numpy as np
import matplotlib.pyplot as plt
import math

if "__main__" == __name__:
    # 分数阶
    alpha = 0.9
    # 步长
    h = 0.005
    # 总点数
    total_points = 601
    t = np.arange(0, total_points * h, h)
    # u存储离散值
    u = np.zeros(total_points)
    # 隐式迭代收敛判断
    epsilon = 0.001
    # 初值
    u[0] = 0.000001
    for n in range(1, total_points):
        # 隐式迭代
        u_n_prev = u[n - 1].copy()
        while True:
            u[n] = math.gamma(2 - alpha) * h ** alpha * 6 * u_n_prev * (n * h) ** (- alpha) / math.gamma(4 - alpha) + (
                    n ** (1 - alpha) - (n - 1) ** (1 - alpha)) * u[0]
            for k in range(1, n):
                u[n] -= ((n - k + 1) ** (1 - alpha) - 2 * (n - k) ** (1 - alpha) + (n - k - 1) ** (1 - alpha)) * u[k]
            if abs(u[n] - u_n_prev) > epsilon:
                u_n_prev = u[n].copy()
            else:
                break
    # 画图
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.plot(t, u, color='r', label='fdm')
    ax.plot(t, t ** 3, color='b', linestyle='--', label='real')
    ax.legend(loc='upper left')
    plt.savefig('./figure/frac_ode_fdm_implicit_01_2d.png')
    plt.show()
