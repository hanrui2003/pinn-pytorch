# -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import math

from scipy.linalg import lstsq
from datetime import datetime

torch.set_default_dtype(torch.float64)
Pi = np.pi
nu = 0.01
v = 0.01
D = 1


def u_real(x, t):
    return np.exp(-t) * np.sin(Pi * (x + t + x * t))


def f_real(x, t):
    u = np.exp(-t) * np.sin(Pi * (x + t + x * t))
    u_x = np.exp(-t) * Pi * (t + 1) * np.cos(Pi * (x + t + x * t))
    u_xx = -np.exp(-t) * (Pi * (t + 1)) ** 2 * np.sin(Pi * (x + t + x * t))
    u_t = np.exp(-t) * (Pi * (x + 1) * np.cos(Pi * (x + t + x * t)) - np.sin(Pi * (x + t + x * t)))
    return u_t + v * u_x - nu * u_xx + D * u


def L_inf_error(v, axis=None):
    return np.max(np.abs(v), axis=axis)


def L_2_error(v, axis=None):
    return np.sqrt(np.sum(np.power(v, 2.0), axis=axis) / v.shape[0])


def weights_init(m):
    rand_mag = 1.0
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if m.weight is not None:
            nn.init.uniform_(m.weight, a=-rand_mag, b=rand_mag)
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-rand_mag, b=rand_mag)


class LocalNet(nn.Module):
    def __init__(self, in_features, hidden_features, x_max, x_min, t_max, t_min):
        super(LocalNet, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        # 半径的倒数
        self.r_inv = torch.tensor([2.0 / (x_max - x_min), 2.0 / (t_max - t_min)])
        # 区域中心
        self.y_c = torch.tensor([(x_max + x_min) / 2, (t_max + t_min) / 2])
        self.phi = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True), nn.Tanh())

    def forward(self, y):
        # 标准化，使得取值在[-1,1]
        y = (y - self.y_c) * self.r_inv
        return self.phi(y)


def pre_define(Nx, Nt, M, X_min, X_max, T_min, T_max):
    print(datetime.now(), "pre_define start")
    models = []
    delta_x = (X_max - X_min) / Nx
    delta_t = (T_max - T_min) / Nt
    for k in range(Nx):
        model_for_x = []
        x_min = k * delta_x + X_min
        x_max = (k + 1) * delta_x + X_min
        for n in range(Nt):
            t_min = n * delta_t + T_min
            t_max = (n + 1) * delta_t + T_min
            model = LocalNet(in_features=2, hidden_features=M, x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max)
            model = model.apply(weights_init)
            for param in model.parameters():
                param.requires_grad = False
            model_for_x.append(model)
        models.append(model_for_x)
    print(datetime.now(), "pre_define end")
    return models


def cal_matrix(models, Nx, Nt, M, Qx, Qt, pde_points, ic_points, bc_points):
    print(datetime.now(), "cal_matrix start")
    # matrix define (Aw=b)
    # pde配点
    A_P = np.zeros((len(pde_points), Nx * Nt * M))

    # 初值配点
    A_I = np.zeros((len(ic_points), Nx * Nt * M))

    # 边界的配点
    A_B = np.zeros((len(bc_points), Nx * Nt * M))

    pde_point = torch.tensor(pde_points, requires_grad=True)
    ic_point = torch.tensor(ic_points, requires_grad=True)
    bc_point = torch.tensor(bc_points, requires_grad=True)

    for k in range(Nx):
        for n in range(Nt):
            M_begin = (k * Nt + n) * M
            pde_out = models[k][n](pde_point)
            pde_values = pde_out.detach().numpy()

            bc_out = models[k][n](bc_point)
            bc_values = bc_out.detach().numpy()

            ic_out = models[k][n](ic_point)
            ic_values = ic_out.detach().numpy()

            grad_u_x = []
            grad_u_t = []
            grad_u_xx = []
            for i in range(M):
                u_x_t = autograd.grad(outputs=pde_out[:, i], inputs=pde_point,
                                      grad_outputs=torch.ones_like(pde_out[:, i]),
                                      create_graph=True, retain_graph=True)[0]
                u_x = u_x_t[:, 0]
                u_t = u_x_t[:, 1]

                u_xx_xt = autograd.grad(outputs=u_x, inputs=pde_point,
                                        grad_outputs=torch.ones_like(u_x),
                                        create_graph=True, retain_graph=True)[0]
                u_xx = u_xx_xt[:, 0]

                grad_u_x.append(u_x.detach().numpy())
                grad_u_t.append(u_t.detach().numpy())
                grad_u_xx.append(u_xx.detach().numpy())

            grad_u_x = np.array(grad_u_x).T
            grad_u_t = np.array(grad_u_t).T
            grad_u_xx = np.array(grad_u_xx).T

            A_P[:, M_begin: M_begin + M] = grad_u_t + v * grad_u_x - nu * grad_u_xx + D * pde_values
            A_I[:, M_begin: M_begin + M] = ic_values
            A_B[:, M_begin: M_begin + M] = bc_values

    A = np.concatenate((A_P, A_I, A_B), axis=0)
    f_P = f_real(pde_points[:, [0]], pde_points[:, [1]])

    interp_point = np.load('convection_diffusion_interp.npz')
    x_interp = interp_point['x_interp']
    y_interp = interp_point['y_interp']
    y_noise = np.interp(ic_points[:, [0]], x_interp, y_interp)
    f_I = u_real(ic_points[:, [0]], ic_points[:, [1]]) + y_noise
    f_B = u_real(bc_points[:, [0]], bc_points[:, [1]])
    f = np.concatenate((f_P, f_I, f_B), axis=0)

    c = 1.0
    # 对每行按其绝对值最大值缩放
    for i in range(len(A)):
        ratio = c / max(-A[i, :].min(), A[i, :].max())
        A[i, :] = A[i, :] * ratio
        f[i] = f[i] * ratio
    print(datetime.now(), "cal_matrix end")
    return A, f


def main(Nx, Nt, M, Qx, Qt):
    print(datetime.now(), "main start")
    x = np.linspace(X_min, X_max, Nx * Qx + 1)
    t = np.linspace(T_min, T_max, Nt * Qt + 1)
    X, T = np.meshgrid(x, t)
    # 用于计算pde损失的点
    pde_points = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    # 带标签值的点，即有真解的点
    ic_points = np.hstack((X[0][:, None], T[0][:, None]))
    left_bc_points = np.hstack((X[:, 0][:, None], T[:, 0][:, None]))
    right_bc_points = np.hstack((X[:, -1][:, None], T[:, -1][:, None]))
    bc_points = np.vstack((left_bc_points, right_bc_points))

    models = pre_define(Nx=Nx, Nt=Nt, M=M, X_min=X_min, X_max=X_max, T_min=T_min, T_max=T_max)

    # matrix define (Aw=b)
    A, f = cal_matrix(models, Nx, Nt, M, Qx, Qt, pde_points, ic_points, bc_points)

    # 为什么选择gelss，默认的不行吗？
    w = lstsq(A, f, lapack_driver="gelss")[0]
    print(datetime.now(), "main process end")

    torch.save(models, 'convection_diffusion_noise_no_psi.pt')
    np.savez('convection_diffusion_noise_no_psi.npz', w=w,
             config=np.array([Nx, Nt, M, Qx, Qt, X_min, X_max, T_min, T_max], dtype=int))

    print(datetime.now(), "main end")


if __name__ == '__main__':
    print(datetime.now(), "Main start")
    # torch.manual_seed(123)
    # np.random.seed(123)

    X_min = 0.0
    X_max = 5.0
    T_min = 0.0
    T_max = 1.0

    # 以下是五组配置，每次训练只取一列
    # x维度划分的区间数
    Nxs = [5, ]
    # t维度划分的区间数
    Nts = [2, ]
    # 每个局部局域的特征函数数量
    Ms = [150, ]
    # x维度每个区间的配点数，Qx+1
    Qxs = [30, ]
    # t维度每个区间的配点数，Qt+1
    Qts = [30, ]

    # # x维度划分的区间数
    # Nxs = [5, ]
    # # t维度划分的区间数
    # Nts = [2, ]
    # # 每个局部局域的特征函数数量
    # Ms = [50, ]
    # # x维度每个区间的配点数，Qx+1
    # Qxs = [10, ]
    # # t维度每个区间的配点数，Qt+1
    # Qts = [10, ]

    loop = zip(Nxs, Nts, Ms, Qxs, Qts)
    for i, item in enumerate(loop):
        Nx, Nt, M, Qx, Qt = item

        main(Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt)

    print(datetime.now(), "Main end")
