import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
from Burgers1D_01 import gp_sample


def plot(X1, T1, U1, X2, T2, U2):
    """
    同时绘制PDE的数值解和真解，上面数值解，下面是真解。
    """
    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')

    cp1 = ax1.contourf(T1, X1, U1, 20, cmap="rainbow")
    fig.colorbar(cp1, ax=ax1)
    ax1.set_title('u(x,t)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')

    ax2.plot_surface(T1, X1, U1, cmap="rainbow")
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u(x,t)')

    cp3 = ax3.contourf(T2, X2, U2, 20, cmap="rainbow")
    fig.colorbar(cp3, ax=ax3)
    ax3.set_title('NN(x,t)')
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')

    ax4.plot_surface(T2, X2, U2, cmap="rainbow")
    ax4.set_xlabel('t')
    ax4.set_ylabel('x')
    ax4.set_zlabel('NN(x,t)')

    plt.show()


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)
    # data = scipy.io.loadmat('./data/Burgers.mat')
    # x = data['x']  # 256 points between -1 and 1 [256x1]
    # t = data['t']  # 100 time points between 0 and 1 [100x1]
    # usol = data['usol'].T  # solution of 256x100 grid points
    #
    # x2 = x[::2]
    # usol2 = usol[:, ::2]
    #
    # Nx = len(x2)
    # Nt = len(t)
    # nu = 0.01 / np.pi
    #
    # dt = t[1] - t[0]
    # h = x2[1] - x2[0]
    # h2 = h ** 2
    # print("grid ratio", nu * dt / h2)
    #
    # D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    # D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    #
    # U = np.zeros((Nt, Nx))
    # U[0] = usol2[0]
    # for n in range(Nt - 1):
    #     u_n = U[n]
    #     u_next = u_n - dt / (2 * h) * D1 @ u_n * u_n + nu * dt / h2 * D2 @ u_n
    #     u_next[np.array([0, -1])] = usol2[n + 1, [0, -1]]
    #     U[n + 1] = u_next
    #
    # X, T = np.meshgrid(x, t)
    # plot(X, T, U, X, T, usol)
    # print()

    test_func = gp_sample(1)[0]
    # x \in [-1, 1], t \in [0, 1]
    Nx = 51
    Nt = 101
    # 初边值点的个数，注意去掉两个交点
    N_icbc = Nx + Nt * 2 - 2
    # 对这些点插值
    icbc = np.linspace(0, 1, N_icbc)
    u_icbc = test_func(icbc)
    # 这些点分给初边值上的点用，u_lb倒个序，方便后续计算
    u_lb = u_icbc[0:Nt][::-1]
    u_ic = u_icbc[Nt - 1:Nt + Nx - 1]
    u_ub = u_icbc[Nt + Nx - 2:]

    x = np.linspace(-1, 1, Nx)
    t = np.linspace(0, 1, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)

    nu = 0.07

    U = np.zeros((Nt, Nx))
    U[0] = u_ic
    for n in range(Nt - 1):
        u_n = U[n]
        u_next = u_n - dt / (2 * h) * D1 @ u_n * u_n + nu * dt / h2 * D2 @ u_n
        u_next[0] = u_lb[n + 1]
        u_next[-1] = u_ub[n + 1]
        U[n + 1] = u_next

    X, T = np.meshgrid(x, t)
    plot(X, T, U, X, T, U)
