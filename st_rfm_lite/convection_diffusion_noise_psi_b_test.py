# -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import math
from convection_diffusion_noise_psi_b import u_real, LocalNet

from scipy.linalg import lstsq
from datetime import datetime
import matplotlib.pyplot as plt


def plot(X1, T1, U1, X2, T2, U2):
    """
    同时绘制PDE的数值解和神经网络解，上面数值解，下面神经网络解。
    """
    min_value = np.min([U1, U2])
    max_value = np.max([U1, U2])

    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(15, 10))
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


def plot_err(X1, T1, U1):
    """
    误差分布
    """
    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    cp1 = ax1.contourf(T1, X1, U1, 20, cmap="rainbow")
    fig.colorbar(cp1, ax=ax1)
    ax1.set_title('err')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')

    ax2.plot_surface(T1, X1, U1, cmap="rainbow")
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('err')

    plt.show()


if __name__ == '__main__':
    print(datetime.now(), "Main start")

    data = np.load("convection_diffusion_noise_psi_b.npz")
    Nx, Nt, M, Qx, Qt, X_min, X_max, T_min, T_max = data['config']
    w = data['w']

    models = torch.load('convection_diffusion_noise_psi_b.pt')

    print(datetime.now(), "test start")
    test_Qx = 2 * Qx
    test_Qt = 2 * Qt

    x = np.linspace(X_min, X_max, Nx * test_Qx + 1)
    t = np.linspace(T_min, T_max, Nt * test_Qt + 1)
    X, T = np.meshgrid(x, t)
    points = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    point = torch.tensor(points, requires_grad=False)
    A = np.zeros((len(point), Nx * Nt * M))
    for k in range(Nx):
        for n in range(Nt):
            M_begin = (k * Nt + n) * M
            out = models[k][n](point)
            values = out.detach().numpy()
            A[:, M_begin: M_begin + M] = values

    numerical_values = np.dot(A, w)
    true_values = u_real(points[:, [0]], points[:, [1]])
    epsilon = np.abs(true_values - numerical_values)

    L_inf = np.max(epsilon)
    L_2 = np.sqrt(np.sum(epsilon ** 2) / len(epsilon))

    print('********************* ERROR *********************')
    print('Nx={:d},Nt={:d},M={:d},Qx={:d},Qt={:d}'.format(Nx, Nt, M, Qx, Qt))
    print(datetime.now(), 'L_inf={:.2e}'.format(L_inf), 'L_2={:.2e}'.format(L_2))
    # print("边值条件误差")
    # print("{:.2e} {:.2e}".format(max(epsilon[0, :]), max(epsilon[-1, :])))
    # print("初值、终值误差")
    # print("{:.2e} {:.2e}".format(max(epsilon[:, 0]), max(epsilon[:, -1])))
    print(datetime.now(), "test end")
    print(datetime.now(), "Main end")

    U_true = true_values.reshape((X.shape[0], X.shape[1]))
    U_numerical = numerical_values.reshape((X.shape[0], X.shape[1]))

    plot(X, T, U_true, X, T, U_numerical)
    # plot_err(X, T, np.abs(U_true - U_numerical))
