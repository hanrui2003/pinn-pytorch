import numpy as np
import torch
from adr_don_ic_01 import gp_sample, ADRNet
from adr_numerical_01 import plot

if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)

    test_ic_func = gp_sample(num=1)[0]

    func_x = np.linspace(0, 1, 100)
    u0_test = test_ic_func(func_x)
    u0_test = np.tile(u0_test, (10000, 1))
    u0_test = torch.from_numpy(u0_test).float()

    x = np.linspace(0, 1, 100)
    t = np.linspace(0, 1, 100)

    X, T = np.meshgrid(x, t)
    y_test = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
    y_test = torch.from_numpy(y_test).float()

    model = torch.load('adr_don_ic_01_gzz.pt', map_location=torch.device('cpu'))

    u_hat = model(u0_test, y_test).detach()

    plot(X, T, u_hat.reshape(100, -1).T)

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

    u = np.zeros((Nx, Nt))
    u[:, 0] = test_ic_func(x)
    for n in range(Nt - 1):
        u_n = u[:, n]
        u_next = u_n + dt * ((D / h2) * D2 @ u_n + k * u_n ** 2)
        u_next[np.array([0, -1])] = 0.
        u[:, n + 1] = u_next

    plot(X, T, u)

    print()
