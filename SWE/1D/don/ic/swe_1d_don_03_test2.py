import numpy as np
import torch
from swe_1d_don_03 import SWENet
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp


def plot(X1, T1, U1, X2, T2, U2):
    """
    同时绘制PDE的数值解和神经网络解，上面数值解，下面神经网络解。
    """
    min_value = np.min([U1, U2])
    max_value = np.max([U1, U2])

    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(14, 10))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')

    cp1 = ax1.contourf(T1, X1, U1, 20, cmap="rainbow", vmin=min_value, vmax=max_value)
    fig.colorbar(cp1, ax=ax1)
    ax1.set_title('u(x,t)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')

    ax2.plot_surface(T1, X1, U1, cmap="rainbow", vmin=min_value, vmax=max_value)
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u(x,t)')
    ax2.set_zlim(min_value, max_value)

    cp3 = ax3.contourf(T2, X2, U2, 20, cmap="rainbow", vmin=min_value, vmax=max_value)
    fig.colorbar(cp3, ax=ax3)
    ax3.set_title('RFM(x,t)')
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')

    ax4.plot_surface(T2, X2, U2, cmap="rainbow", vmin=min_value, vmax=max_value)
    ax4.set_xlabel('t')
    ax4.set_ylabel('x')
    ax4.set_zlabel('RFM(x,t)')
    ax4.set_zlim(min_value, max_value)

    plt.show()


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

    model = torch.load('swe_1d_don_03_e-6.pt', map_location=torch.device('cpu'))
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
    u0_test = torch.from_numpy(u0_nn).float()

    X, T = np.meshgrid(x_nn, t_nn)
    x_test = X.flatten()[:, None]
    t_test = T.flatten()[:, None]
    y_test = np.hstack((x_test, t_test))
    y_test = torch.from_numpy(y_test).float()
    predict = model(u0_test, y_test).detach().numpy()
    h_hat = predict[:, 0].reshape(-1, N_x_nn)

    # 数值解
    step = (N_x - 1) // (N_x_nn - 1)
    idx_filter = [i for i in range(0, N_x, step)]

    u1 = rk4(u0, N_t - 1, dx, dt)
    h1 = u1[:, :N_x]
    h1 = h1[idx_filter, :][:, idx_filter]

    epsilon = np.abs(h1 - h_hat)

    L_inf = np.max(epsilon)
    L_2 = np.sqrt(np.sum(epsilon ** 2) / len(epsilon))

    print('L_inf={:.2e}'.format(L_inf), 'L_2={:.2e}'.format(L_2))

    plot(X, T, h_hat, X, T, h1)
