# -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import math
from convection_diffusion_2d_da_no_psi import u_real, LocalNet

from scipy.linalg import lstsq
from datetime import datetime
import matplotlib.pyplot as plt


def plot(X, Y, U1, U2):
    """
    同时绘制PDE的数值解和神经网络解，上面数值解，下面神经网络解。
    """
    min_value = np.min([U1, U2])
    max_value = np.max([U1, U2])
    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(231, projection='3d')
    ax2 = fig.add_subplot(232, projection='3d')
    ax3 = fig.add_subplot(233, projection='3d')
    ax4 = fig.add_subplot(234, projection='3d')
    ax5 = fig.add_subplot(235, projection='3d')
    ax6 = fig.add_subplot(236, projection='3d')

    ax1.plot_surface(X, Y, U1[0], cmap="rainbow", vmin=min_value, vmax=max_value)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlim(min_value, max_value)

    ax2.plot_surface(X, Y, U1[1], cmap="rainbow", vmin=min_value, vmax=max_value)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlim(min_value, max_value)

    ax3.plot_surface(X, Y, U1[2], cmap="rainbow", vmin=min_value, vmax=max_value)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlim(min_value, max_value)

    ax4.plot_surface(X, Y, U2[0], cmap="rainbow", vmin=min_value, vmax=max_value)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlim(min_value, max_value)

    ax5.plot_surface(X, Y, U2[1], cmap="rainbow", vmin=min_value, vmax=max_value)
    ax5.set_xlabel('x')
    ax5.set_ylabel('y')
    ax5.set_zlim(min_value, max_value)

    ax6.plot_surface(X, Y, U2[2], cmap="rainbow", vmin=min_value, vmax=max_value)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_zlim(min_value, max_value)

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

    data = np.load("convection_diffusion_2d_da_no_psi_1.npz")
    Nx, Ny, Nt, M, Qx, Qy, Qt, X_min, X_max, Y_min, Y_max, T_min, T_max = data['config']
    w = data['w']

    models = torch.load('convection_diffusion_2d_da_no_psi_1.pt')

    print(datetime.now(), "test start")
    test_Qx = 2 * Qx
    test_Qy = 2 * Qy

    t_slice = np.linspace(T_min, T_max, 3)

    x = np.linspace(X_min, X_max, Nx * test_Qx + 1)
    y = np.linspace(Y_min, Y_max, Ny * test_Qy + 1)
    X, Y = np.meshgrid(x, y)
    point_xy = np.hstack((X.flatten()[:, None], Y.flatten()[:, None]))

    U_true = []
    U_numerical = []
    for t in t_slice:
        t_column = t * np.ones((point_xy.shape[0], 1))
        points = np.hstack((point_xy, t_column))
        point = torch.tensor(points, requires_grad=False)
        A = np.zeros((len(point), Nx * Ny * Nt * M))
        for k in range(Nx):
            for j in range(Ny):
                for n in range(Nt):
                    M_begin = (k * Ny * Nt + j * Nt + n) * M
                    out = models[k][j][n](point)
                    values = out.detach().numpy()
                    A[:, M_begin: M_begin + M] = values

        numerical_values = np.dot(A, w)
        true_values = u_real(points[:, [0]], points[:, [1]], points[:, [2]])

        epsilon = np.abs(true_values - numerical_values)
        L_inf = np.max(epsilon)
        L_2 = np.sqrt(np.sum(epsilon ** 2) / len(epsilon))
        print('Error for t :', t)
        print('Nx={:d},Ny={:d},M={:d},Qx={:d},Qy={:d}'.format(Nx, Ny, M, test_Qx, test_Qy))
        print(datetime.now(), 'L_inf={:.2e}'.format(L_inf), 'L_2={:.2e}'.format(L_2))
        U_true.append(true_values.reshape((X.shape[0], X.shape[1])))
        U_numerical.append(numerical_values.reshape((X.shape[0], X.shape[1])))

    print(datetime.now(), "Main end")
    plot(X, Y, U_true, U_numerical)
