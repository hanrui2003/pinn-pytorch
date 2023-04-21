import numpy as np
import torch
from adr_don_ic_01 import gp_sample, ADRNet
import matplotlib.pyplot as plt


def plot(X1, T1, U1, X2, T2, U2):
    """
    同时绘制PDE的数值解和神经网络解，上面数值解，下面神经网络解。
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
    # torch.manual_seed(1234)
    # np.random.seed(1234)

    # 测试的初值函数（随机生成）
    test_ic_func = gp_sample(num=1)[0]

    # 以下是数值解
    Nx = 50
    Nt = 100
    D = 0.01
    k = 0.01

    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2
    print("grid ratio", D * dt / h2)

    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    X, T = np.meshgrid(x, t)

    U1 = np.zeros((Nx, Nt))
    U1[:, 0] = test_ic_func(x)
    for n in range(Nt - 1):
        u_n = U1[:, n]
        u_next = u_n + dt * ((D / h2) * D2 @ u_n + k * u_n ** 2)
        u_next[np.array([0, -1])] = 0.
        U1[:, n + 1] = u_next

    # 以下是神经网络解
    func_x = np.linspace(0, 1, 100)
    u0_test = test_ic_func(func_x)
    u0_test = np.tile(u0_test, (Nx * Nt, 1))
    u0_test = torch.from_numpy(u0_test).float()

    y_test = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    y_test = torch.from_numpy(y_test).float()

    model = torch.load('adr_don_ic_01_gzz_02.pt', map_location=torch.device('cpu'))

    u_hat = model(u0_test, y_test).detach()

    plot(X, T, U1.T, X, T, u_hat.reshape(100, -1))

    u_truth = U1.flatten(order="F")[:, None]
    u_pred = u_hat.numpy()

    # 计算相对误差，(真值-预测值)的范数/真值的范数
    error = np.linalg.norm(u_truth - u_pred) / np.linalg.norm(u_truth)
    print("error", error)
