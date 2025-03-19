# -*- coding: utf-8 -*-
"""
用于EAJAM
"""
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import math
from scipy.sparse import csr_matrix

from scipy.linalg import lstsq
from scipy.sparse.linalg import lsmr
from scipy.sparse.linalg import lsqr
from datetime import datetime

torch.set_default_dtype(torch.float64)
Pi = np.pi
nu = 0.01


def u_real(x, t):
    u0 = 2 * np.cos(Pi * x + Pi / 5) + 3 / 2 * np.cos(2 * Pi * x - 3 * Pi / 5)
    u1 = 2 * np.cos(Pi * t + Pi / 5) + 3 / 2 * np.cos(2 * Pi * t - 3 * Pi / 5)

    u = u0 * u1

    return u


def f_real(x, t):
    u0 = 2 * np.cos(Pi * x + Pi / 5) + 3 / 2 * np.cos(2 * Pi * x - 3 * Pi / 5)
    u1 = 2 * np.cos(Pi * t + Pi / 5) + 3 / 2 * np.cos(2 * Pi * t - 3 * Pi / 5)

    u1t = -2 * Pi * np.sin(Pi * t + Pi / 5) - 3 * Pi * np.sin(2 * Pi * t - 3 * Pi / 5)
    ut = u0 * u1t

    u0xx = -2 * Pi * Pi * np.cos(Pi * x + Pi / 5) - 6 * Pi * Pi * np.cos(2 * Pi * x - 3 * Pi / 5)
    uxx = u0xx * u1

    f = ut - nu * uxx

    return f


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
    def __init__(self, in_features, hidden_features, x_max, x_min, t_max, t_min, Nx, Nt, n_x, n_t):
        super(LocalNet, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        # 基函数个数
        self.M = M
        self.Nx = Nx
        self.Nt = Nt
        self.n_x = n_x
        self.n_t = n_t
        # 半径的倒数
        self.r_inv = torch.tensor([2.0 / (x_max - x_min), 2.0 / (t_max - t_min)])
        # 区域中心
        self.y_c = torch.tensor([(x_max + x_min) / 2, (t_max + t_min) / 2])
        self.phi = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True), nn.Tanh())

    def forward(self, y):
        # 标准化，使得取值在[-1,1]
        y = (y - self.y_c) * self.r_inv
        z = self.phi(y)
        x = y[:, [0]]
        t = y[:, [1]]
        if self.n_x == 0:
            psi_x = ((x >= -1) & (x < 3 / 4)) * 1.0 + \
                    ((x >= 3 / 4) & (x < 5 / 4)) * (1 - torch.sin(2 * np.pi * x)) / 2
        elif self.n_x == self.Nx - 1:
            psi_x = ((x >= -5 / 4) & (x < -3 / 4)) * (1 + torch.sin(2 * np.pi * x)) / 2 + \
                    ((x >= -3 / 4) & (x <= 1)) * 1.0
        else:
            psi_x = ((x >= -5 / 4) & (x < -3 / 4)) * (1 + torch.sin(2 * np.pi * x)) / 2 + \
                    ((x >= -3 / 4) & (x < 3 / 4)) * 1.0 + \
                    ((x >= 3 / 4) & (x < 5 / 4)) * (1 - torch.sin(2 * np.pi * x)) / 2

        if self.n_t == 0:
            psi_t = ((t >= -1) & (t < 3 / 4)) * 1.0 + \
                    ((t >= 3 / 4) & (t < 5 / 4)) * (1 - torch.sin(2 * np.pi * t)) / 2
        elif self.n_t == self.Nt - 1:
            psi_t = ((t >= -5 / 4) & (t < -3 / 4)) * (1 + torch.sin(2 * np.pi * t)) / 2 + \
                    ((t >= -3 / 4) & (t <= 1)) * 1.0
        else:
            psi_t = ((t >= -5 / 4) & (t < -3 / 4)) * (1 + torch.sin(2 * np.pi * t)) / 2 + \
                    ((t >= -3 / 4) & (t < 3 / 4)) * 1.0 + \
                    ((t >= 3 / 4) & (t < 5 / 4)) * (1 - torch.sin(2 * np.pi * t)) / 2

        return psi_x * psi_t * z


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
            model = LocalNet(in_features=2, hidden_features=M, x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max,
                             Nx=Nx, Nt=Nt, n_x=k, n_t=n)
            model = model.apply(weights_init)
            for param in model.parameters():
                param.requires_grad = False
            model_for_x.append(model)
        models.append(model_for_x)
    print(datetime.now(), "pre_define end")
    return models


def cal_matrix(models, Nx, Nt, M, Qx, Qt, pde_points, label_points, initial=None):
    print(datetime.now(), "cal_matrix start")
    # matrix define (Aw=b)
    # pde配点
    A_P = np.zeros((len(pde_points), Nx * Nt * M))

    # 带标签的配点
    A_L = np.zeros((len(label_points), Nx * Nt * M))

    pde_point = torch.tensor(pde_points, requires_grad=True)
    label_point = torch.tensor(label_points, requires_grad=True)

    for k in range(Nx):
        for n in range(Nt):
            M_begin = (k * Nt + n) * M
            # u_t - laplacian u= f(x, t) on omega [0, T]
            pde_out = models[k][n](pde_point)
            pde_values = pde_out.detach().numpy()

            label_out = models[k][n](label_point)
            label_values = label_out.detach().numpy()

            # u_t - laplacian u= f(x, t) on omega [0, T]
            grads = []
            grad_u_t = []
            grad_u_xx = []
            for i in range(M):
                u_x_t = autograd.grad(outputs=pde_out[:, i], inputs=pde_point,
                                      grad_outputs=torch.ones_like(pde_out[:, i]),
                                      create_graph=True, retain_graph=True)[0]

                # 这里为什么不能一并求二阶导u_xx和u_tt，分开求和一并求，debug的结果确实不一样，待研究
                g_2_x = autograd.grad(outputs=u_x_t[:, 0], inputs=pde_point,
                                      grad_outputs=torch.ones_like(u_x_t[:, 0]),
                                      create_graph=True, retain_graph=True)[0]
                # g_2_t = autograd.grad(outputs=u_x_t[:, 1], inputs=pde_point,
                #                       grad_outputs=torch.ones_like(u_x_t[:, 1]),
                #                       create_graph=True, retain_graph=True)[0]

                u_t = u_x_t[:, 1]
                u_xx = g_2_x[:, 0]

                grad_u_t.append(u_t.detach().numpy())
                grad_u_xx.append(u_xx.detach().numpy())

            grad_u_t = np.array(grad_u_t).T
            grad_u_xx = np.array(grad_u_xx).T

            A_P[:, M_begin: M_begin + M] = grad_u_t - nu * grad_u_xx

            A_L[:, M_begin: M_begin + M] = label_values

    A = np.concatenate((A_P, A_L), axis=0)
    f_P = f_real(pde_points[:, [0]], pde_points[:, [1]])
    f_L = u_real(label_points[:, [0]], label_points[:, [1]])
    f = np.concatenate((f_P, f_L), axis=0)
    print(datetime.now(), "cal_matrix end")

    return A, f


def main(Nx, Nt, M, Qx, Qt):
    x = np.linspace(X_min, X_max, Nx * Qx + 1)
    t = np.linspace(T_min, T_max, Nt * Qt + 1)
    X, T = np.meshgrid(x, t)
    # 用于计算pde损失的点
    pde_points = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    # 带标签值的点，即有真解的点
    ic_points = np.hstack((X[0][:, None], T[0][:, None]))
    left_bc_points = np.hstack((X[:, 0][:, None], T[:, 0][:, None]))
    right_bc_points = np.hstack((X[:, -1][:, None], T[:, -1][:, None]))
    label_points = np.vstack((ic_points, left_bc_points, right_bc_points))

    models = pre_define(Nx=Nx, Nt=Nt, M=M, X_min=X_min, X_max=X_max, T_min=T_min, T_max=T_max)

    true_values = list()
    numerical_values = list()
    L_is = list()
    L_2s = list()

    # matrix define (Aw=b)
    A, f = cal_matrix(models, Nx, Nt, M, Qx, Qt, pde_points, label_points)
    # 矩阵的缩放不做也没关系？
    c = 100.0
    # 对每行按其绝对值最大值缩放
    # 为什么不按照绝对值的最大值缩放？这样会映射到[-c,c]，岂不完美？看文档应该是绝对值，代码错误？还有就是最大值有极小的概率是0，也是个风险点。
    for i in range(len(A)):
        ratio = c / max(-A[i, :].min(), A[i, :].max())
        A[i, :] = A[i, :] * ratio
        f[i] = f[i] * ratio

    # A_sparse = csr_matrix(A)
    # # 计算稀疏矩阵的实际内存占用
    # data_size = A_sparse.data.nbytes  # 非零元素的字节数
    # indices_size = A_sparse.indices.nbytes  # 列索引的字节数
    # indptr_size = A_sparse.indptr.nbytes  # 行偏移的字节数
    # total_memory_bytes = data_size + indices_size + indptr_size  # 总内存占用（字节）
    # total_memory_mb = total_memory_bytes / (1024 * 1024)  # 转换为 MB
    # print("A sparse memory size : ", total_memory_mb)
    # print("A dense memory size : ", A.nbytes / (1024 * 1024))

    # 记录训练开始时间
    start_time = datetime.now()
    print("Training started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    w = lstsq(A, f, lapack_driver="gelss")[0]

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)

    torch.save(models, 'oned_rfm_diff_psi_b.pt')
    np.savez('oned_rfm_diff_psi_b.npz', w=w,
             config=np.array([Nx, Nt, M, Qx, Qt, X_min, X_max, T_min, T_max], dtype=int))


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
    Ms = [200, ]
    # x维度每个区间的配点数，Qx+1
    Qxs = [30, ]
    # t维度每个区间的配点数，Qt+1
    Qts = [30, ]

    # # x维度划分的区间数
    # Nxs = [5, ]
    # # t维度划分的区间数
    # Nts = [2, ]
    # # 每个局部局域的特征函数数量
    # Ms = [3, ]
    # # x维度每个区间的配点数，Qx+1
    # Qxs = [4, ]
    # # t维度每个区间的配点数，Qt+1
    # Qts = [4, ]

    loop = zip(Nxs, Nts, Ms, Qxs, Qts)
    for i, item in enumerate(loop):
        Nx, Nt, M, Qx, Qt = item

        main(Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt)

    print(datetime.now(), "Main end")
