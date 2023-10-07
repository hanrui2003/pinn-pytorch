import numpy as np
import torch
from swe_1d_don_tbc_ic_03 import SWENet
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp


def dudt_of_shallow_water(t, w):
    g = 100
    n = len(w) // 2
    h = w[:n]
    u = w[n:] / h
    F1 = h * u
    F2 = h * u * u + h ** 2 / 2 * g
    J = np.abs(u) + np.sqrt(h * g)
    J_interface = np.maximum(J[:-1], J[1:])
    w1 = h
    F1_interface = (F1[:-1] + F1[1:]) / 2 + J_interface * (w1[:-1] - w1[1:]) / 2
    w2 = h * u
    F2_interface = (F2[:-1] + F2[1:]) / 2 + J_interface * (w2[:-1] - w2[1:]) / 2
    F1_interface = np.concatenate([[0], F1_interface, [0]])
    F2_interface = np.concatenate([[h[0] ** 2 / 2 * g], F2_interface, [h[-1] ** 2 / 2 * g]])
    return np.concatenate([F1_interface[:-1] - F1_interface[1:],
                           F2_interface[:-1] - F2_interface[1:]]) / dx


def solve2(u0, t):
    res = solve_ivp(dudt_of_shallow_water, t_span=(0, t[-1]), y0=u0, t_eval=t)
    return res['y'].T


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

    model = torch.load('swe_1d_don_tbc_ic_03_e-6.pt', map_location=torch.device('cpu'))
    print("model", model)

    N_x = 201
    N_t = 201

    x = np.linspace(0, 1, N_x)
    t = np.linspace(0, 1, N_t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    # [0,0.1)随机数
    mu = 0.1 * np.random.random()

    h0 = 0.1 + 0.1 * np.exp(-64 * (x - mu) ** 2)
    m0 = np.zeros(N_x)
    u0 = np.hstack((h0, m0))

    # 神经网络
    N_x_nn = 101
    x_nn = np.linspace(0, 1, N_x_nn)
    t_nn = np.linspace(0, 1, 101)

    h0_nn = 0.1 + 0.1 * np.exp(-64 * (x_nn - mu) ** 2)
    v0_nn = np.zeros(N_x_nn)
    u0_nn = np.hstack((h0_nn, v0_nn))

    X, T = np.meshgrid(x_nn, t_nn)
    x_test = X.flatten()[:, None]
    t_test = T.flatten()[:, None]
    y_test = np.hstack((x_test, t_test))

    # 拼接输入
    u0_nn = np.repeat(u0_nn[None, :], len(y_test), axis=0)
    z_test = np.hstack((y_test, u0_nn))

    z_test = torch.from_numpy(z_test).float()
    predict = model(z_test).detach().numpy()
    h_hat = predict[:, 0].reshape(-1, N_x_nn)

    # 数值解
    step = (N_x - 1) // (N_x_nn - 1)
    idx_filter = [i for i in range(0, N_x, step)]

    u1 = rk4(u0, N_t - 1, dx, dt)
    h1 = u1[:, :N_x]
    h1 = h1[idx_filter, :][:, idx_filter]

    u2 = solve2(u0, np.linspace(0, 0.1, N_t))
    h2 = u2[:, :N_x]
    h2 = h2[idx_filter, :][:, idx_filter]

    # 创建图表对象
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # 初始化曲线
    line1, = ax.plot([], [], color="red", linestyle='-', label="t=")
    line2, = ax.plot([], [], color="blue", linestyle='--', label="r_err=")
    line3, = ax.plot([], [], color="green", linestyle='--', label="r_err=")


    # 更新函数
    def update(i):
        line1.set_data(x_nn, h_hat[i])
        line2.set_data(x_nn, h1[i])
        line3.set_data(x_nn, h2[i])
        line1.set_label("t=" + str(round(i * 0.01, 2)))
        line2.set_label("err=" + str(round(np.linalg.norm(h1[i] - h_hat[i]) / np.linalg.norm(h1[i]), 5)))
        line3.set_label("err=" + str(round(np.linalg.norm(h2[i] - h_hat[i]) / np.linalg.norm(h2[i]), 5)))
        ax.legend()
        return line1, line2, line3,


    # 创建动画
    anim = animation.FuncAnimation(fig, update, frames=len(h_hat), interval=100)
    # 保存动画
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
                                         codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format("swe_1d_don_tbc_ic_03"), writer=mpeg_writer)

    plt.show()
