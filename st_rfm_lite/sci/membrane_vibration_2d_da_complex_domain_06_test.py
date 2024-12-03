# -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import math
from membrane_vibration_2d_da_complex_domain_06 import u_real, LocalNet, pick_point, circles

from scipy.linalg import lstsq
from datetime import datetime
import matplotlib.pyplot as plt


def plot(X, Y, U):
    """
    同时绘制PDE的数值解和神经网络解，上面数值解，下面神经网络解。
    """
    min_value = np.nanmin(U)
    max_value = np.nanmax(U)
    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(10, 8))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223, projection='3d')
    ax4 = fig.add_subplot(224, projection='3d')

    ax1.plot_surface(X, Y, U[0], cmap="rainbow", vmin=min_value, vmax=max_value)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('u')
    ax1.set_zlim(min_value, max_value)

    ax2.plot_surface(X, Y, U[1], cmap="rainbow", vmin=min_value, vmax=max_value)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('u')
    ax2.set_zlim(min_value, max_value)

    ax3.plot_surface(X, Y, U[2], cmap="rainbow", vmin=min_value, vmax=max_value)
    ax3.set_xlabel('x')
    ax3.set_ylabel('y')
    ax3.set_zlabel('u')
    ax3.set_zlim(min_value, max_value)

    ax4.plot_surface(X, Y, U[3], cmap="rainbow", vmin=min_value, vmax=max_value)
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('u')
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

    data = np.load("membrane_vibration_2d_da_complex_domain_06_50.npz")
    Nx, Ny, Nt, M, Qx, Qy, Qt, X_min, X_max, Y_min, Y_max, T_min, T_max = data['config']
    w = data['w']

    models = torch.load('membrane_vibration_2d_da_complex_domain_06_50.pt')

    print(datetime.now(), "test start")
    test_Qx = 2 * Qx
    test_Qy = 2 * Qy
    test_Qt = 2 * Qt

    x = np.linspace(X_min, X_max, Nx * test_Qx + 1)
    y = np.linspace(Y_min, Y_max, Ny * test_Qy + 1)
    t = np.linspace(T_min, T_max, Nt * test_Qt + 1)

    X, Y = np.meshgrid(x, y)

    # 初始化一个布尔数组，用于标记在圆外的点
    outside = np.ones_like(X, dtype=bool)

    # 对每个圆，标记出在圆内的点
    for circle in circles:
        x0, y0 = circle['center']
        r = circle['radius']
        distances = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        outside = outside & (distances >= r)

    point_xy = np.vstack((X[outside], Y[outside])).T

    U_true = []
    U_numerical = []
    for t in [.5, 1., 1.5, 2.]:
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

        u_numerical = np.full_like(X, np.nan)
        u_numerical[outside] = numerical_values.flatten()
        U_numerical.append(u_numerical)

    # 计算全部误差 start
    all_points = pick_point(x, y, t)
    all_point = torch.tensor(all_points, requires_grad=False)
    A = np.zeros((len(all_point), Nx * Ny * Nt * M))
    for k in range(Nx):
        for j in range(Ny):
            for n in range(Nt):
                M_begin = (k * Ny * Nt + j * Nt + n) * M
                out = models[k][j][n](all_point)
                values = out.detach().numpy()
                A[:, M_begin: M_begin + M] = values

    all_numerical_values = np.dot(A, w)
    all_true_values = u_real(all_points[:, [0]], all_points[:, [1]], all_points[:, [2]])

    all_epsilon = np.abs(all_true_values - all_numerical_values)
    L_inf_all = np.max(all_epsilon)
    L_2_all = np.sqrt(np.sum(all_epsilon ** 2) / len(all_epsilon))
    print(
        'Nx={:d},Ny={:d},Nt={:d},M={:d},test_Qx={:d},test_Qy={:d},test_Qt={:d}'.format(Nx, Ny, Nt, M, test_Qx, test_Qy,
                                                                                       test_Qt))
    print(datetime.now(), 'L_inf_all={:.2e}'.format(L_inf_all), 'L_2_all={:.2e}'.format(L_2_all))

    # 计算全部误差 end

    print(datetime.now(), "Main end")
    # plot(X, Y, U_numerical)
