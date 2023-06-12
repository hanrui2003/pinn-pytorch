import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from swe_1d_pinn_01 import SWENet


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
    torch.manual_seed(123)
    np.random.seed(123)

    model = torch.load('swe_1d_pinn_01_2_5e-6.pt', map_location=torch.device('cpu'))
    print("model", model)

    N_x = 101
    N_t = 201

    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 2, 201)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    X, T = np.meshgrid(x, t)
    x_test = X.flatten()[:, None]
    t_test = T.flatten()[:, None]
    y_test = np.hstack((x_test, t_test))
    y_test = torch.from_numpy(y_test).float()
    predict = model(y_test).detach().numpy()
    h_hat = predict[:, 0].reshape(-1, 101)

    # 数值解
    h0 = 0.1 + 0.1 * np.exp(-64 * (x - 0.25) ** 2)
    m0 = np.zeros(N_x)
    u0 = np.hstack((h0, m0))
    u = rk4(u0, N_t - 1, dx, dt)
    h = u[:, :N_x]

    # 创建图表对象
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # 初始化曲线
    line1, = ax.plot([], [], color="red", linestyle='-', label="t=")
    line2, = ax.plot([], [], color="blue", linestyle='--', label="t=")


    # 更新函数
    def update(i):
        line1.set_data(x, h_hat[i])
        line2.set_data(x, h[i])
        line1.set_label("t=" + str(round(i * 0.01, 2)))
        ax.legend()
        return line1,


    # 创建动画
    anim = animation.FuncAnimation(fig, update, frames=len(h_hat), interval=100)
    # 保存动画
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
                                         codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format("swe_1d_pinn_01_2"), writer=mpeg_writer)

    plt.show()

    print()
