# -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import math
from oned_diff_pinn import u_real, DiffNet, X_min, X_max, T_min, T_max, Nx, Nt, M, Qx, Qt

from datetime import datetime
import matplotlib.pyplot as plt


def plot(X1, T1, U1, X2, T2, U2):
    """
    同时绘制PDE的数值解和神经网络解，上面数值解，下面神经网络解。
    """
    min_value = np.min([U1, U2])
    max_value = np.max([U1, U2])
    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(221)
    # ax1.set_ylim
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
    ax3.set_title('PINN(x,t)')
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')

    ax4.plot_surface(T2, X2, U2, cmap="rainbow", vmin=min_value, vmax=max_value)
    ax4.set_xlabel('t')
    ax4.set_ylabel('x')
    ax4.set_zlabel('PINN(x,t)')
    ax4.set_zlim(min_value, max_value)

    plt.show()


if __name__ == '__main__':
    print(datetime.now(), "Main start")
    model = torch.load('oned_diff_pinn_01.pt', map_location=torch.device('cpu'))

    test_Qx = 2 * Qx
    test_Qt = 2 * Qt

    x = np.linspace(X_min, X_max, Nx * test_Qx + 1)
    t = np.linspace(T_min, T_max, Nt * test_Qt + 1)
    X, T = np.meshgrid(x, t)
    points = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    point = torch.tensor(points, dtype=torch.float, requires_grad=False)
    numerical_values = model(point).detach().numpy()
    U_numerical = numerical_values.reshape((X.shape[0], X.shape[1]))
    U_true = u_real(X, T)
    epsilon = np.abs(U_true - U_numerical)
    L_inf = np.max(epsilon)
    L_2 = np.sqrt(np.sum(epsilon ** 2) / epsilon.size)

    print('********************* ERROR *********************')
    print('Nx={:d},Nt={:d},M={:d},Qx={:d},Qt={:d}'.format(Nx, Nt, M, Qx, Qt))
    print(datetime.now(), 'L_inf={:.2e}'.format(L_inf), 'L_2={:.2e}'.format(L_2))
    print(datetime.now(), "Main end")

    # plot(X, T, U_true, X, T, U_numerical)
