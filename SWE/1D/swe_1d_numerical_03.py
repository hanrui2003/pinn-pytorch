import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

"""
每次一秒钟，然后以最后一次的高度和零动量作为初值，继续迭代
理论上和一次性迭代应该有区别，因为一次性迭代动量是不为0的
"""


# initial height profile
def init_u(x):
    return 0.1 + 0.1 * np.exp(-64 * (x - 0.25) ** 2)


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
    L = 1
    N_x = 100
    dx = L / N_x

    # time parameter
    t0 = 0.0
    T = 5

    # initial height vector
    ini_h = np.array([init_u((i + 0.5) * dx) for i in range(N_x)])
    # ini_h_left = 0.2 * np.ones(20)
    # ini_h_right = 0.1 * np.ones(80)
    # ini_h = np.hstack((ini_h_left, ini_h_right))
    # initial velocity vector
    ini_m = np.array([0 for i in range(N_x)])

    u0 = np.hstack((ini_h, ini_m))

    u_all = np.empty((1, N_x * 2))
    u_all[0] = u0
    for _ in range(5):
        u = rk4(u0, 100, dx, 0.01)
        h = u[:, :N_x]
        m = np.zeros_like(h)
        u_new = np.hstack((h, m))
        u_all = np.vstack((u_all, u_new[1:, :]))
        u0 = u_new[-1, :]

    h = u_all[:, :100]

    # 创建图表对象
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # 初始化曲线
    line, = ax.plot([], [], label="t=")

    x = np.linspace(0.01, 1, 100)


    # 更新函数
    def update(i):
        line.set_data(x, h[i])
        line.set_label("t=" + str(round(i * 0.01, 2)))
        ax.legend()
        return line,


    # 创建动画
    anim = animation.FuncAnimation(fig, update, frames=len(h), interval=20)
    # 保存动画
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
                                         codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format("swe_1d_numerical_03"), writer=mpeg_writer)

    plt.show()

    print()
