# -*- coding: utf-8 -*-
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
import math
import sympy as sp

from scipy.linalg import lstsq
from datetime import datetime

"""
去掉初值条件，即lambda_i=0
"""

torch.set_default_dtype(torch.float64)
X_min = 0.0
X_max = 4.0
Y_min = 0.0
Y_max = 4.0
T_min = 0.0
T_max = 2.0
mu = 2 * np.pi / (X_max - X_min)
nu = 2 * np.pi / (Y_max - Y_min)
lambda_ = np.sqrt(mu ** 2 + nu ** 2)
# 陡度因子
alpha = 0.1
# 波速
c = 1.0

# 定义三个圆的参数（圆心和半径）
circles = [
    {'center': (1, 1), 'radius': 0.5},
    {'center': (3, 2), 'radius': 0.8},
    {'center': (2, 3.5), 'radius': 0.7}
]


def pick_point(x, y, t):
    X, Y, T = np.meshgrid(x, y, t)
    # 初始化一个布尔数组，用于标记在圆外的点
    outside = np.ones_like(X, dtype=bool)

    for circle in circles:
        x0, y0 = circle['center']
        r = circle['radius']
        distances = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        outside = outside & (distances >= r)

    # 筛选出位于所有圆外的点，并组成 (m, 2) 的点列
    remaining_points = np.vstack((X[outside], Y[outside], T[outside])).T
    return remaining_points


# 定义符号变量
x, y, t = sp.symbols('x y t')

x0, y0 = circles[0]['center']
r0 = circles[0]['radius']
x1, y1 = circles[1]['center']
r1 = circles[1]['radius']
x2, y2 = circles[2]['center']
r2 = circles[2]['radius']
chi = (1 - sp.exp(-alpha * ((x - x0) ** 2 + (y - y0) ** 2 - r0 ** 2) / r0 ** 2)) * \
      (1 - sp.exp(-alpha * ((x - x1) ** 2 + (y - y1) ** 2 - r1 ** 2) / r1 ** 2)) * \
      (1 - sp.exp(-alpha * ((x - x2) ** 2 + (y - y2) ** 2 - r2 ** 2) / r2 ** 2))

u = chi * sp.sin(mu * x) * sp.sin(nu * y) * (2 * sp.cos(lambda_ * t) + sp.sin(lambda_ * t))
u_xx = sp.diff(u, x, 2)
u_yy = sp.diff(u, y, 2)
u_tt = sp.diff(u, t, 2)
f = u_tt - c * (u_xx + u_yy)

u_real = sp.lambdify((x, y, t), u, 'numpy')
f_real = sp.lambdify((x, y, t), f, 'numpy')


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
    def __init__(self, in_features, hidden_features, x_max, x_min, y_max, y_min, t_max, t_min):
        super(LocalNet, self).__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        # 半径的倒数
        self.r_inv = torch.tensor([2.0 / (x_max - x_min), 2.0 / (y_max - y_min), 2.0 / (t_max - t_min)])
        # 区域中心
        self.z_c = torch.tensor([(x_max + x_min) / 2, (y_max + y_min) / 2, (t_max + t_min) / 2])
        self.phi = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True), nn.Tanh())

    def forward(self, z):
        # 标准化，使得取值在[-1,1]
        z = (z - self.z_c) * self.r_inv
        return self.phi(z)


def pre_define(Nx, Ny, Nt, M, X_min, X_max, Y_min, Y_max, T_min, T_max):
    print(datetime.now(), "pre_define start")
    models = []
    delta_x = (X_max - X_min) / Nx
    delta_y = (Y_max - Y_min) / Ny
    delta_t = (T_max - T_min) / Nt
    for k in range(Nx):
        model_for_x = []
        x_min = k * delta_x + X_min
        x_max = (k + 1) * delta_x + X_min
        for j in range(Ny):
            model_for_xy = []
            y_min = j * delta_y + Y_min
            y_max = (j + 1) * delta_y + Y_min
            for n in range(Nt):
                t_min = n * delta_t + T_min
                t_max = (n + 1) * delta_t + T_min
                model = LocalNet(in_features=3, hidden_features=M, x_min=x_min, x_max=x_max, y_min=y_min, y_max=y_max,
                                 t_min=t_min, t_max=t_max)
                model = model.apply(weights_init)
                for param in model.parameters():
                    param.requires_grad = False
                model_for_xy.append(model)

            model_for_x.append(model_for_xy)

        models.append(model_for_x)
    print(datetime.now(), "pre_define end")
    return models


def cal_matrix(models, Nx, Ny, Nt, M, Qx, Qy, Qt, pde_points, ic_points, bc_points, obs_points):
    print(datetime.now(), "cal_matrix start")
    # matrix define (Aw=b)
    # pde配点
    A_P = np.zeros((len(pde_points), Nx * Ny * Nt * M))

    # 初值配点
    A_I = np.zeros((len(ic_points), Nx * Ny * Nt * M))

    # 边界的配点
    A_B = np.zeros((len(bc_points), Nx * Ny * Nt * M))

    # 观测点
    A_O = np.zeros((len(obs_points), Nx * Ny * Nt * M))

    pde_point = torch.tensor(pde_points, requires_grad=True)
    ic_point = torch.tensor(ic_points, requires_grad=True)
    bc_point = torch.tensor(bc_points, requires_grad=True)
    obs_point = torch.tensor(obs_points, requires_grad=True)

    for k in range(Nx):
        for j in range(Ny):
            for n in range(Nt):
                M_begin = (k * Ny * Nt + j * Nt + n) * M
                pde_out = models[k][j][n](pde_point)
                pde_values = pde_out.detach().numpy()

                bc_out = models[k][j][n](bc_point)
                bc_values = bc_out.detach().numpy()

                ic_out = models[k][j][n](ic_point)
                ic_values = ic_out.detach().numpy()

                obs_out = models[k][j][n](obs_point)
                obs_values = obs_out.detach().numpy()

                grad_u_xx = []
                grad_u_yy = []
                grad_u_tt = []
                for i in range(M):
                    u_x_y_t = autograd.grad(outputs=pde_out[:, i], inputs=pde_point,
                                            grad_outputs=torch.ones_like(pde_out[:, i]),
                                            create_graph=True, retain_graph=True)[0]
                    u_x = u_x_y_t[:, 0]
                    u_y = u_x_y_t[:, 1]
                    u_t = u_x_y_t[:, 2]

                    u_xx_xy_xt = autograd.grad(outputs=u_x, inputs=pde_point,
                                               grad_outputs=torch.ones_like(u_x),
                                               create_graph=True, retain_graph=True)[0]
                    u_xx = u_xx_xy_xt[:, 0]

                    u_yx_yy_yt = autograd.grad(outputs=u_y, inputs=pde_point,
                                               grad_outputs=torch.ones_like(u_y),
                                               create_graph=True, retain_graph=True)[0]
                    u_yy = u_yx_yy_yt[:, 1]

                    u_tx_ty_tt = autograd.grad(outputs=u_t, inputs=pde_point,
                                               grad_outputs=torch.ones_like(u_t),
                                               create_graph=True, retain_graph=True)[0]
                    u_tt = u_tx_ty_tt[:, 2]

                    grad_u_xx.append(u_xx.detach().numpy())
                    grad_u_yy.append(u_yy.detach().numpy())
                    grad_u_tt.append(u_tt.detach().numpy())

                grad_u_xx = np.array(grad_u_xx).T
                grad_u_yy = np.array(grad_u_yy).T
                grad_u_tt = np.array(grad_u_tt).T

                A_P[:, M_begin: M_begin + M] = grad_u_tt - c * (grad_u_xx + grad_u_yy)
                A_I[:, M_begin: M_begin + M] = ic_values
                A_B[:, M_begin: M_begin + M] = bc_values
                A_O[:, M_begin: M_begin + M] = obs_values

    f_P = f_real(pde_points[:, [0]], pde_points[:, [1]], pde_points[:, [2]])
    f_I = u_real(ic_points[:, [0]], ic_points[:, [1]], ic_points[:, [2]])
    f_B = u_real(bc_points[:, [0]], bc_points[:, [1]], bc_points[:, [2]])
    f_O = u_real(obs_points[:, [0]], obs_points[:, [1]], obs_points[:, [2]])

    lambda_p = 1.0
    lambda_i = 0.0
    lambda_b = 1.0
    lambda_o = 1.0

    A_I *= lambda_i
    f_I *= lambda_i

    A_B *= lambda_b
    f_B *= lambda_b

    A = np.concatenate((A_P, A_I, A_B, A_O), axis=0)
    f = np.concatenate((f_P, f_I, f_B, f_O), axis=0)

    print(datetime.now(), "cal_matrix end")
    return A, f


def main(Nx, Ny, Nt, M, Qx, Qy, Qt):
    print(datetime.now(), "main start")
    x = np.linspace(X_min, X_max, Nx * Qx + 1)
    y = np.linspace(Y_min, Y_max, Ny * Qy + 1)
    t = np.linspace(T_min, T_max, Nt * Qt + 1)
    X, Y, T = np.meshgrid(x, y, t)
    # 用于计算pde损失的点
    pde_points = pick_point(x, y, t)
    # 带标签值的点，即有真解的点
    ic_points = pick_point(x, y, 0)
    top_bc_points = np.hstack((X[0].flatten()[:, None], Y[0].flatten()[:, None], T[0].flatten()[:, None]))
    bottom_bc_points = np.hstack((X[-1].flatten()[:, None], Y[-1].flatten()[:, None], T[-1].flatten()[:, None]))
    left_bc_points = np.hstack(
        (X[:, 0, :].flatten()[:, None], Y[:, 0, :].flatten()[:, None], T[:, 0, :].flatten()[:, None]))
    right_bc_points = np.hstack(
        (X[:, -1, :].flatten()[:, None], Y[:, -1, :].flatten()[:, None], T[:, -1, :].flatten()[:, None]))
    bc_points = np.vstack((top_bc_points, bottom_bc_points, left_bc_points, right_bc_points))

    t_obs = np.linspace(T_min, T_max, 5)[1:]
    x_obs = np.linspace(X_min, X_max, 9)[1:-1]
    y_obs = np.linspace(Y_min, Y_max, 9)[1:-1]
    obs_points = pick_point(x_obs, y_obs, t_obs)

    models = pre_define(Nx=Nx, Ny=Ny, Nt=Nt, M=M, X_min=X_min, X_max=X_max, Y_min=Y_min, Y_max=Y_max, T_min=T_min,
                        T_max=T_max)

    # matrix define (Aw=b)
    A, f = cal_matrix(models, Nx, Ny, Nt, M, Qx, Qy, Qt, pde_points, ic_points, bc_points, obs_points)

    # 为什么选择gelss，默认的不行吗？
    w, residuals, *_ = lstsq(A, f, lapack_driver="gelss")
    print(datetime.now(), "main process end,", "shape of A :", A.shape, "residuals :", residuals, "mse : ",
          residuals / len(A), "L_2 error :", np.sqrt(residuals / len(A)))

    torch.save(models, 'membrane_vibration_2d_da_complex_domain_06_' + str(M) + '.pt')
    np.savez('membrane_vibration_2d_da_complex_domain_06_' + str(M) + '.npz', w=w,
             config=np.array([Nx, Ny, Nt, M, Qx, Qy, Qt, X_min, X_max, Y_min, Y_max, T_min, T_max], dtype=int))

    print(datetime.now(), "main end")


if __name__ == '__main__':
    print(datetime.now(), "Main start")
    # torch.manual_seed(123)
    # np.random.seed(123)

    # 以下是五组配置，每次训练只取一列
    # x维度划分的区间数
    Nxs = [4, ]
    # y维度划分的区间数
    Nys = [4, ]
    # t维度划分的区间数
    Nts = [2, ]
    # 每个局部局域的特征函数数量
    Ms = [50, ]
    # x维度每个区间的配点数，Qx+1
    Qxs = [10, ]
    # y维度每个区间的配点数，Qx+1
    Qys = [10, ]
    # t维度每个区间的配点数，Qt+1
    Qts = [10, ]

    loop = zip(Nxs, Nys, Nts, Ms, Qxs, Qys, Qts)
    for i, item in enumerate(loop):
        Nx, Ny, Nt, M, Qx, Qy, Qt = item
        print("Nx=", Nx, ",Ny=", Ny, ",Nt=", Nt, ",M=", M, ",Qx=", Qx, ",Qy=", Qy, ",Qt=", Qt)
        main(Nx=Nx, Ny=Ny, Nt=Nt, M=M, Qx=Qx, Qy=Qy, Qt=Qt)

    print(datetime.now(), "Main end")
