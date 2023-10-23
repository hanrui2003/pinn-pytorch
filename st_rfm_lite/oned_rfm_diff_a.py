# -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
import torch.nn as nn

from scipy.linalg import lstsq
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
    def __init__(self, in_features, hidden_features, x_max, x_min, t_max, t_min):
        super(LocalNet, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        # 基函数个数
        self.M = M
        # 半径的倒数
        self.r_inv = torch.tensor([2.0 / (x_max - x_min), 2.0 / (t_max - t_min)])
        # 区域中心
        self.x_c = torch.tensor([(x_max + x_min) / 2, (t_max + t_min) / 2])
        self.phi = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True), nn.Tanh())

    def forward(self, x):
        x = self.r_inv * (x - self.x_c)
        x = self.phi(x)
        return x


def pre_define_rfm(Nx, Nt, M, Qx, Qt, X_min, X_max, T_min, T_max):
    models = []
    points = []
    delta_x = (X_max - X_min) / Nx
    delta_t = (T_max - T_min) / Nt
    for k in range(Nx):
        model_for_x = []
        point_for_x = []
        x_min = k * delta_x + X_min
        x_max = (k + 1) * delta_x + X_min
        x_divide = np.linspace(x_min, x_max, Qx + 1)
        for n in range(Nt):
            t_min = n * delta_t + T_min
            t_max = (n + 1) * delta_t + T_min
            model = LocalNet(in_features=2, hidden_features=M, x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max)
            model = model.apply(weights_init)
            for param in model.parameters():
                param.requires_grad = False
            model_for_x.append(model)
            t_divide = np.linspace(t_min, t_max, Qt + 1)
            # 对x和t做笛卡尔积，然后转成三维数组，本质就是这个区域的配点
            grid = np.array(list(itertools.product(x_divide, t_divide))).reshape(Qx + 1, Qt + 1, 2)
            point_for_x.append(torch.tensor(grid, requires_grad=True))
        models.append(model_for_x)
        points.append(point_for_x)
    return models, points


def cal_matrix(models, points, Nx, Nt, M, Qx, Qt):
    # matrix define (Aw=b)
    # pde配点
    A_P = np.zeros([Nx * Nt * Qx * Qt, Nx * Nt * M])  # u_t - lapacian u - u^2 = f(x, t) on omega [0, T]
    f_P = np.zeros([Nx * Nt * Qx * Qt, 1])

    # 左边界配点
    A_B_L = np.zeros([Nt * Qt, Nx * Nt * M])  # u(x,t) = g(x, t) on x=0
    f_B_L = np.zeros([Nt * Qt, 1])

    # 右边界配点
    A_B_R = np.zeros([Nt * Qt, Nx * Nt * M])  # u(x,t) = g(x, t) on x=1
    f_B_R = np.zeros([Nt * Qt, 1])

    # 初值配点
    A_I = np.zeros([Nx * Qx, Nx * Nt * M])  # u(x,t) = h(x) on t=0
    f_I = np.zeros([Nx * Qx, 1])

    C_0x = np.zeros([(Nx - 1) * Qt * Nt, Nx * Nt * M])  # C^0 continuity on x
    f_c_0x = np.zeros([(Nx - 1) * Qt * Nt, 1])

    C_0t = np.zeros([(Nt - 1) * Qx * Nx, Nx * Nt * M])  # C^0 continuity on t
    f_c_0t = np.zeros([(Nt - 1) * Qx * Nx, 1])

    C_1x = np.zeros([(Nx - 1) * Qt * Nt, Nx * Nt * M])  # C^1 continuity on x
    f_c_1x = np.zeros([(Nx - 1) * Qt * Nt, 1])

    for k in range(Nx):
        for n in range(Nt):
            # u_t - laplacian u= f(x, t) on omega [0, T]
            in_ = points[k][n].detach().numpy()
            out = models[k][n](points[k][n])
            values = out.detach().numpy()
            M_begin = (k * Nt + n) * M

            # 初值处理 u(x, 0) = ...
            if n == 0:
                f_I[k * Qx: (k + 1) * Qx, :] = u_real(in_[:Qx, 0, 0], in_[:Qx, 0, 1]).reshape((Qx, 1))

            # 左边值处理 u(0,t) = ..
            if k == 0:
                # A_2[0, M_begin : M_begin + M] = values[0,:Qt,:]
                f_B_L[n * Qt: (n + 1) * Qt, :] = u_real(in_[0, :Qt, 0], in_[0, :Qt, 1]).reshape((Qt, 1))

            # # 右边值处理 u(1,t) = ..
            if k == Nx - 1:
                # A_5[n*Qt : n*Qt+Qt, M_begin : M_begin + M] = values[-1,:Qt,:]
                f_B_R[n * Qt: (n + 1) * Qt, :] = u_real(in_[-1, :Qt, 0], in_[-1, :Qt, 1]).reshape((Qt, 1))

            # 转成二维点列，去除区域的右边界和上边界的点。然后计算方程的右端项f
            f_in = in_[:Qx, :Qt, :].reshape((-1, 2))
            f_P[(k * Nt + n) * Qx * Qt: (k * Nt + n + 1) * Qx * Qt, :] = f_real(f_in[:, 0], f_in[:, 1]).reshape(-1, 1)

            # u_t - laplacian u= f(x, t) on omega [0, T]
            grads = []
            grads_2_xx = []
            for i in range(M):
                g_1 = torch.autograd.grad(outputs=out[:, :, i], inputs=points[k][n],
                                          grad_outputs=torch.ones_like(out[:, :, i]),
                                          create_graph=True, retain_graph=True)[0]
                grads.append(g_1.squeeze().detach().numpy())

                g_2_x = torch.autograd.grad(outputs=g_1[:, :, 0], inputs=points[k][n],
                                            grad_outputs=torch.ones_like(out[:, :, i]),
                                            create_graph=True, retain_graph=True)[0]
                grads_2_xx.append(g_2_x[:, :, 0].squeeze().detach().numpy())

            grads = np.array(grads).swapaxes(0, 3)

            grads_t = grads[1, :Qx, :Qt, :]
            grads_t = grads_t.reshape(-1, M)
            grads_2_xx = np.array(grads_2_xx)
            grads_2_xx = grads_2_xx[:, :Qx, :Qt]
            grads_2_xx = grads_2_xx.transpose(1, 2, 0).reshape(-1, M)

            A_P[(k * Nt + n) * Qx * Qt: (k * Nt + n + 1) * Qx * Qt, M_begin: M_begin + M] = grads_t - nu * grads_2_xx

            # u(0,t) = ..
            if k == 0:
                A_B_L[n * Qt: (n + 1) * Qt, M_begin: M_begin + M] = values[0, :Qt, :]
            # u(1,t) = ..
            if k == Nx - 1:
                A_B_R[n * Qt: (n + 1) * Qt, M_begin: M_begin + M] = values[-1, :Qt, :]
            # u(x,0) = ..
            if n == 0:
                A_I[k * Qx: (k + 1) * Qx, M_begin: M_begin + M] = values[:Qx, 0, :]

            # C0 continuity on t
            if n > 0:
                C_0t[(n - 1) * Qx * Nx + k * Qx: (n - 1) * Qx * Nx + k * Qx + Qx, M_begin: M_begin + M] = values[:Qx, 0,
                                                                                                          :]
            if n < Nt - 1:
                C_0t[n * Qx * Nx + k * Qx: n * Qx * Nx + k * Qx + Qx, M_begin: M_begin + M] = -values[:Qx, -1, :]

            # C0 continuity on x
            if k > 0:
                C_0x[(k - 1) * Qt * Nt + n * Qt: (k - 1) * Qt * Nt + (n + 1) * Qt, M_begin: M_begin + M] = values[0,
                                                                                                           :Qt, :]
            if k < Nx - 1:
                C_0x[k * Nt * Qt + n * Qt: k * Nt * Qt + (n + 1) * Qt, M_begin: M_begin + M] = -values[-1, :Qt, :]

            # C1 continuity on x
            if k > 0:
                C_1x[(k - 1) * Qt * Nt + n * Qt: (k - 1) * Qt * Nt + (n + 1) * Qt, M_begin: M_begin + M] = grads[0, 0,
                                                                                                           :Qt, :]
            if k < Nx - 1:
                C_1x[k * Nt * Qt + n * Qt: k * Nt * Qt + (n + 1) * Qt, M_begin: M_begin + M] = -grads[0, -1, :Qt, :]

    A = np.concatenate((A_P, A_B_L, A_B_R, A_I, C_0t, C_0x, C_1x), axis=0)
    f = np.concatenate((f_P, f_B_L, f_B_R, f_I, f_c_0t, f_c_0x, f_c_1x), axis=0)
    return A, f


def main(Nx, Nt, M, Qx, Qt):
    # prepare models and collocation points
    models, points = pre_define_rfm(Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt, X_min=X_min, X_max=X_max, T_min=T_min, T_max=T_max)

    true_values = list()
    numerical_values = list()
    L_is = list()
    L_2s = list()

    # matrix define (Aw=b)
    A, f = cal_matrix(models, points, Nx, Nt, M, Qx, Qt)
    # 矩阵的缩放不做也没关系？
    c = 100.0
    # 对每行按其绝对值最大值缩放
    # 为什么不按照绝对值的最大值缩放？这样会映射到[-c,c]，岂不完美？看文档应该是绝对值，代码错误？还有就是最大值有极小的概率是0，也是个风险点。
    for i in range(len(A)):
        ratio = c / max(-A[i, :].min(), A[i, :].max())
        A[i, :] = A[i, :] * ratio
        f[i] = f[i] * ratio
    # 为什么选择gelss，默认的不行吗？
    w = lstsq(A, f, lapack_driver="gelss")[0]

    true_values_, numerical_values_, L_i, L_2 = test(models=models,
                                                     Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt,
                                                     w=w, X_min=X_min, X_max=X_max, T_min=T_min, T_max=T_max)

    true_values.append(true_values_)
    numerical_values.append(numerical_values_)
    L_is.append(L_i)
    L_2s.append(L_2)

    true_values = np.concatenate(true_values, axis=1)[:, ::1]
    numerical_values = np.concatenate(numerical_values, axis=1)[:, ::1]
    e_i = L_inf_error(true_values - numerical_values, axis=0).reshape(1, Nt, -1)
    e_2 = L_2_error(true_values - numerical_values, axis=0).reshape(1, Nt, -1)
    L_is = np.array(L_is)
    L_2s = np.array(L_2s)

    return L_is, L_2s, e_i, e_2


def test(models, Nx=1, Nt=1, M=1, Qx=1, Qt=1, w=None, X_min=0, X_max=1, T_min=0, T_max=1):
    delta_x = (X_max - X_min) / Nx
    delta_t = (T_max - T_min) / Nt

    epsilon = []
    true_values = []
    numerical_values = []

    test_Qx = 2 * Qx
    test_Qt = 2 * Qt

    for k in range(Nx):
        epsilon_x = []
        true_value_x = []
        numerical_value_x = []
        for n in range(Nt):
            # forward and grad
            x_min = k * delta_x + X_min
            x_max = (k + 1) * delta_x + X_min
            x_divide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
            t_min = n * delta_t
            t_max = (n + 1) * delta_t
            t_divide = np.linspace(t_min, t_max, test_Qt + 1)[:test_Qt]
            grid = np.array(list(itertools.product(x_divide, t_divide))).reshape(test_Qx, test_Qt, 2)
            test_point = torch.tensor(grid, requires_grad=False)
            in_ = test_point.detach().numpy()
            true_value = u_real(in_[:, :, 0], in_[:, :, 1])
            values = models[k][n](test_point).detach().numpy()
            numerical_value = np.dot(values, w[k * Nt * M + n * M: k * Nt * M + n * M + M, :]).reshape(test_Qx, test_Qt)
            e = np.abs(true_value - numerical_value)
            true_value_x.append(true_value)
            numerical_value_x.append(numerical_value)
            epsilon_x.append(e)
        epsilon_x = np.concatenate(epsilon_x, axis=1)
        epsilon.append(epsilon_x)
        true_value_x = np.concatenate(true_value_x, axis=1)
        true_values.append(true_value_x)
        numerical_value_x = np.concatenate(numerical_value_x, axis=1)
        numerical_values.append(numerical_value_x)
    epsilon = np.concatenate(epsilon, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    numerical_values = np.concatenate(numerical_values, axis=0)
    e = epsilon.reshape((-1, 1))
    L_i = L_inf_error(e)
    L_2 = L_2_error(e)
    print('********************* ERROR *********************')
    print('Nx={:d},Nt={:d},M={:d},Qx={:d},Qt={:d}'.format(Nx, Nt, M, Qx, Qt))
    print(datetime.now(), 'L_inf={:.2e}'.format(L_i), 'L_2={:.2e}'.format(L_2))
    print("边值条件误差", "{:.2e} {:.2e}".format(max(epsilon[0, :]), max(epsilon[-1, :])))
    print("初值、终值误差", "{:.2e} {:.2e}".format(max(epsilon[:, 0]), max(epsilon[:, -1])))

    return true_values, numerical_values, L_i, L_2


if __name__ == '__main__':
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
    Nts = [5, ]
    # 每个局部局域的特征函数数量
    Ms = [300, ]
    # x维度每个区间的配点数，Qx+1
    Qxs = [30, ]
    # t维度每个区间的配点数，Qt+1
    Qts = [30, ]

    # # x维度划分的区间数
    # Nxs = [5, ]
    # # t维度划分的区间数
    # Nts = [2, ]
    # # 每个局部局域的特征函数数量
    # Ms = [30, ]
    # # x维度每个区间的配点数，Qx+1
    # Qxs = [6, ]
    # # t维度每个区间的配点数，Qt+1
    # Qts = [6, ]

    loop = zip(Nxs, Nts, Ms, Qxs, Qts)
    for i, item in enumerate(loop):
        Nx, Nt, M, Qx, Qt = item

        L_i_res = list()
        L_2_res = list()
        e_i_res = list()
        e_2_res = list()
        result = []

        L_is, L_2s, e_i, e_2 = main(Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt)

        L_i_res.append(L_is)
        L_2_res.append(L_2s)
        e_i_res.append(np.stack(e_i, axis=0))
        e_2_res.append(np.stack(e_2, axis=0))

        L_i = L_inf_error(L_is)
        L_2 = L_2_error(L_2s)

        result.append([L_i, L_2])

        L_i_res = np.array(L_i_res)
        L_2_res = np.array(L_2_res)
        e_i_res = np.array(e_i_res).reshape(1, 1, Nt, -1)
        e_2_res = np.array(e_2_res).reshape(1, 1, Nt, -1)

        result = np.array(result)
