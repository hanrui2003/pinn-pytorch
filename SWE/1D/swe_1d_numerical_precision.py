import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from matplotlib import animation

"""
数据区间[0,1]*[0,0.5]
"""


def dudt(u_n, dx):
    # 波速
    c = 0.2
    # 重力加速度
    g = 1.0
    length = len(u_n)
    half_len = length // 2
    curr_dhdt = np.zeros(half_len)
    curr_dmdt = np.zeros(half_len)
    curr_h = u_n[:half_len]
    curr_m = u_n[half_len:]
    for i in range(half_len):
        if i == 0:
            curr_dhdt[i] = (-curr_m[i] - curr_m[i + 1] + c * (curr_h[i] - 2 * curr_h[i] + curr_h[i + 1])) / (2 * dx)
            curr_dmdt[i] = ((curr_m[i] ** 2 / curr_h[i] - curr_m[i + 1] ** 2 / curr_h[i + 1]) + (
                    curr_h[i] ** 2 - curr_h[i + 1] ** 2) * g / 2 + c * (
                                    -curr_m[i] - 2 * curr_m[i] + curr_m[i + 1])) / (2 * dx)
        elif i == half_len - 1:
            curr_dhdt[i] = (curr_m[i - 1] + curr_m[i] + c * (curr_h[i - 1] - 2 * curr_h[i] + curr_h[i])) / (2 * dx)
            curr_dmdt[i] = ((curr_m[i - 1] ** 2 / curr_h[i - 1] - curr_m[i] ** 2 / curr_h[i]) + (
                    curr_h[i - 1] ** 2 - curr_h[i] ** 2) * g / 2 + c * (
                                    curr_m[i - 1] - 2 * curr_m[i] - curr_m[i])) / (2 * dx)
        else:
            curr_dhdt[i] = (curr_m[i - 1] - curr_m[i + 1] + c * (curr_h[i - 1] - 2 * curr_h[i] + curr_h[i + 1])) / (
                    2 * dx)
            curr_dmdt[i] = ((curr_m[i - 1] ** 2 / curr_h[i - 1] - curr_m[i + 1] ** 2 / curr_h[i + 1]) + (
                    curr_h[i - 1] ** 2 - curr_h[i + 1] ** 2) * g / 2 + c * (
                                    curr_m[i - 1] - 2 * curr_m[i] + curr_m[i + 1])) / (2 * dx)

    return np.hstack((curr_dhdt, curr_dmdt))


def rk4(u0, n, dx, dt):
    """

    :param u0: 初值
    :param n: 迭代次数
    :param dt: 时间步长
    :return: 数值解
    """
    u = np.zeros((n + 1, len(u0)))
    u[0] = u0
    for i in range(n):
        k1 = dudt(u[i], dx)
        k2 = dudt(u[i] + 0.5 * dt * k1, dx)
        k3 = dudt(u[i] + 0.5 * dt * k2, dx)
        k4 = dudt(u[i] + dt * k3, dx)
        u[i + 1] = u[i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return u


if "__main__" == __name__:
    # 插值点个数
    N_interp = 1001
    # 扰动项，防止后续的 cholesky 分解过程中，有些元素非常小甚至为零时，导致平方根运算不稳定。
    jitter = 1e-10
    x_interp = np.linspace(0, 1, N_interp)
    l = 0.05
    K = np.exp(-0.5 * (x_interp[:, None] - x_interp) ** 2 / (l ** 2))
    # cholesky 分解，得到下三角矩阵
    L = np.linalg.cholesky(K + jitter * np.eye(N_interp))
    y_interp = np.dot(L, np.random.randn(N_interp))
    y_max = y_interp.max()
    y_min = y_interp.min()
    y_interp = 0.1 + 0.1 * (y_interp - y_min) / (y_max - y_min)

    # 数值解配置1
    N_x1 = 201
    N_t1 = 2001
    print("N_x1 :", N_x1, "N_t1 :", N_t1)

    x1 = np.linspace(0, 1, N_x1)
    t1 = np.linspace(0, 10, N_t1)
    dx1 = x1[1] - x1[0]
    dt1 = t1[1] - t1[0]

    h0_1 = np.interp(x1, x_interp, y_interp)
    m0_1 = np.zeros_like(h0_1)
    u0_1 = np.hstack((h0_1, m0_1))

    u1 = rk4(u0_1, N_t1 - 1, dx1, dt1)
    h1 = u1[:, :N_x1]

    # 数值解配置2
    N_x2 = 401
    N_t2 = 4001
    print("N_x2 :", N_x2, "N_t2 :", N_t2)

    x2 = np.linspace(0, 1, N_x2)
    t2 = np.linspace(0, 10, N_t2)
    dx2 = x2[1] - x2[0]
    dt2 = t2[1] - t2[0]

    h0_2 = np.interp(x2, x_interp, y_interp)
    m0_2 = np.zeros_like(h0_2)
    u0_2 = np.hstack((h0_2, m0_2))

    u2 = rk4(u0_2, N_t2 - 1, dx2, dt2)
    h2 = u2[:, :N_x2]
    t_filter = [i * 2 for i in range(N_t1)]
    x_filter = [i * 2 for i in range(N_x1)]
    h2 = h2[t_filter, :][:, x_filter]

    # 创建图表对象
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # 初始化曲线
    line1, = ax.plot([], [], color="red", linestyle='-', label="grid=" + str(N_x1))
    line2, = ax.plot([], [], color="blue", linestyle='--', label="grid=" + str(N_x2))


    # 更新函数
    def update(i):
        line1.set_data(x1, h1[i])
        line2.set_data(x1, h2[i])
        # plt.text(1, 3, "err=" + str(round(np.linalg.norm(h2[i] - h1[i]) / np.linalg.norm(h1[i]), 5)), ha='center',
        #          va='center')
        line2.set_label("err=" + str(round(np.linalg.norm(h1[i] - h2[i]) / np.linalg.norm(h1[i]), 5)))
        ax.legend()
        return line1, line2,


    # 创建动画
    anim = animation.FuncAnimation(fig, update, frames=len(h1), interval=20)
    # 保存动画
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
                                         codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format("swe_1d_numerical_precision"), writer=mpeg_writer)

    plt.show()
