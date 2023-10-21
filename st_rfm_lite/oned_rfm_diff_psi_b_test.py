# -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import math
from oned_rfm_diff_psi_b import u_real, LocalNet

from scipy.linalg import lstsq
from datetime import datetime

if __name__ == '__main__':
    print(datetime.now(), "Main start")

    data = np.load("oned_rfm_diff_psi_b.npz")
    Nx, Nt, M, Qx, Qt, X_min, X_max, T_min, T_max = data['config']
    w = data['w']

    models = torch.load('oned_rfm_diff_psi_b.pt')

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
    # 就是取绝对值操作
    epsilon = np.maximum(epsilon, -epsilon)

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
