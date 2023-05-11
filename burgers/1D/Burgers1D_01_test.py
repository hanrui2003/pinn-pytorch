import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
from Burgers1D_01 import gp_sample, BurgersNet


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
    test_func = gp_sample(1)[0]
    # x \in [-1, 1], t \in [0, 1]
    Nx = 61
    Nt = 201
    nu = 0.07

    x = np.linspace(-1, 1, Nx)
    t = np.linspace(0, 1, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2
    print("grid ratio", nu * dt / h2)

    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)

    # 插值的时候，初边值点的个数，注意去掉两个交点
    N_icbc = Nx + Nt * 2 - 2
    # 对这些点插值
    icbc = np.linspace(0, 1, N_icbc)
    u_icbc = test_func(icbc)

    # 这些点分给初边值上的点用，u_lb倒个序，方便后续计算
    u_lb = u_icbc[0:Nt][::-1]
    u_ic = u_icbc[Nt - 1:Nt + Nx - 1]
    u_ub = u_icbc[Nt + Nx - 2:]

    U = np.zeros((Nt, Nx))
    U[0] = u_ic
    for n in range(Nt - 1):
        u_n = U[n]
        u_next = u_n - dt / (2 * h) * D1 @ u_n * u_n + nu * dt / h2 * D2 @ u_n
        u_next[0] = u_lb[n + 1]
        u_next[-1] = u_ub[n + 1]
        U[n + 1] = u_next

    # 选取观测点的子网格，就是对大网格取子矩阵，这里要与训练时保持一致
    # x和t每隔多少个点取一个点
    x_interval = (Nx - 1) // 20
    t_interval = (Nt - 1) // 10
    idx_x = [i for i in range(0, Nx, x_interval)]
    idx_t = [i for i in range(0, Nt, t_interval)]

    # numpy的meshgrid参数，第一个是横轴，正方向向右，第二个是纵轴，正方向向下，坐标原点左上角；
    X, T = np.meshgrid(x, t)

    y_test = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    y_test = torch.from_numpy(y_test).float()

    o_test = U[idx_t][:, idx_x].flatten()
    o_test = np.tile(o_test, (Nx * Nt, 1))
    o_test = torch.from_numpy(o_test).float()

    model = torch.load('Burgers1D_01_gzz_0.01.pt', map_location=torch.device('cpu'))
    print("model", model)

    u_hat = model(o_test, y_test).detach()

    u_truth = U.flatten(order="C")[:, None]
    u_pred = u_hat.numpy()

    max_error = max(abs(u_truth - u_pred))
    print("max_error", max_error)

    # 计算相对误差，(真值-预测值)的范数/真值的范数
    error = np.linalg.norm(u_truth - u_pred) / np.linalg.norm(u_truth)
    print("error", error)

    plot(X, T, U, X, T, u_hat.reshape(Nt, -1))
