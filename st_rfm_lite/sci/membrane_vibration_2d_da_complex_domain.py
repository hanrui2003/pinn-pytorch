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
alpha = 10

# 定义三个圆的参数（圆心和半径）
Circles = [
    {'center': (1, 1), 'radius': 0.5},
    {'center': (3, 2), 'radius': 0.8},
    {'center': (2, 3.5), 'radius': 0.7}
]


def pick_point(x, y, t):
    X, Y, T = np.meshgrid(x, y, t)
    # 初始化一个布尔数组，用于标记在圆外的点
    outside = np.ones_like(X, dtype=bool)

    for circle in Circles:
        x0, y0 = circle['center']
        r = circle['radius']
        distances = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
        outside = outside & (distances >= r)

    # 筛选出位于所有圆外的点，并组成 (m, 2) 的点列
    remaining_points = np.vstack((X[outside], Y[outside], T[outside])).T
    return remaining_points


def u_e(x, y, t):
    return np.sin(mu * x) * np.sin(nu * y) * (2 * np.cos(lambda_ * t) + np.sin(lambda_ * t))


# 定义 u_e 的二阶导数
def laplacian_u_e(x, y, t):
    u_e_value = u_e(x, y, t)
    return -lambda_ ** 2 * u_e_value


# 定义光滑函数
def chi(x, y, x_center, y_center, radius):
    return 1 - np.exp(-alpha * ((x - x_center) ** 2 + (y - y_center) ** 2 - radius ** 2) / radius ** 2)


def chi0(x, y):
    x_center, y_center = Circles[0]['center']
    radius = Circles[0]['radius']
    return chi(x, y, x_center, y_center, radius)


def chi1(x, y):
    x_center, y_center = Circles[1]['center']
    radius = Circles[1]['radius']
    return chi(x, y, x_center, y_center, radius)


def chi2(x, y):
    x_center, y_center = Circles[2]['center']
    radius = Circles[2]['radius']
    return chi(x, y, x_center, y_center, radius)


# 定义总的光滑函数（乘积）
def multiple_chi(x, y, circles):
    result = 1
    for circle in circles:
        x_center, y_center = circle['center']
        radius = circle['radius']
        result *= chi(x, y, x_center, y_center, radius)
    return result


# 计算光滑函数 chi 的导数
def chi_x(x, y, x_center, y_center, radius):
    exp_term = np.exp(-alpha * ((x - x_center) ** 2 + (y - y_center) ** 2 - radius ** 2) / radius ** 2)
    return (2 * alpha * (x - x_center) / radius ** 2) * exp_term


def chi_y(x, y, x_center, y_center, radius):
    exp_term = np.exp(-alpha * ((x - x_center) ** 2 + (y - y_center) ** 2 - radius ** 2) / radius ** 2)
    return (2 * alpha * (y - y_center) / radius ** 2) * exp_term


def chi_xx(x, y, x_center, y_center, radius):
    exp_term = np.exp(-alpha * ((x - x_center) ** 2 + (y - y_center) ** 2 - radius ** 2) / radius ** 2)
    return (2 * alpha / radius ** 2 - 4 * alpha ** 2 * (x - x_center) ** 2 / radius ** 4) * exp_term


def chi_yy(x, y, x_center, y_center, radius):
    exp_term = np.exp(-alpha * ((x - x_center) ** 2 + (y - y_center) ** 2 - radius ** 2) / radius ** 2)
    return (2 * alpha / radius ** 2 - 4 * alpha ** 2 * (y - y_center) ** 2 / radius ** 4) * exp_term


# multiple_chi_x: 计算多个 chi 的一阶 x 导数
def multiple_chi_x(x, y, circles):
    result = 0
    for circle in circles:
        x_center, y_center = circle['center']
        radius = circle['radius']

        # 当前 chi 的一阶 x 导数
        chi_x_i = chi_x(x, y, x_center, y_center, radius)

        # 更新乘积（去掉当前 chi）
        for other_circle in circles:
            if other_circle != circle:
                other_x_center, other_y_center = other_circle['center']
                other_radius = other_circle['radius']
                chi_x_i *= chi(x, y, other_x_center, other_y_center, other_radius)

        # 累加结果
        result += chi_x_i
    return result


# multiple_chi_y: 计算多个 chi 的一阶 y 导数
def multiple_chi_y(x, y, circles):
    result = 0
    for circle in circles:
        x_center, y_center = circle['center']
        radius = circle['radius']

        # 当前 chi 的一阶 y 导数
        chi_y_i = chi_y(x, y, x_center, y_center, radius)

        # 更新乘积（去掉当前 chi）
        for other_circle in circles:
            if other_circle != circle:
                other_x_center, other_y_center = other_circle['center']
                other_radius = other_circle['radius']
                chi_y_i *= chi(x, y, other_x_center, other_y_center, other_radius)

        # 累加结果
        result += chi_y_i
    return result


# multiple_chi_xx: 计算多个 chi 的二阶 x 导数
def multiple_chi_xx(x, y, circles):
    result = 0
    for i, circle in enumerate(circles):
        x_center, y_center = circle['center']
        radius = circle['radius']

        # 当前 chi 的二阶 x 导数
        chi_xx_i = chi_xx(x, y, x_center, y_center, radius)

        # 当前 chi 的一阶 x 导数
        chi_x_i = chi_x(x, y, x_center, y_center, radius)

        # 计算其他 chi 的乘积
        product_of_other_chis = 1
        sum_of_other_chi_xx_terms = 0
        sum_of_cross_terms = 0

        for j, other_circle in enumerate(circles):
            if i != j:
                other_x_center, other_y_center = other_circle['center']
                other_radius = other_circle['radius']

                # 其他 chi 的一阶和二阶导数
                chi_x_other = chi_x(x, y, other_x_center, other_y_center, other_radius)
                chi_xx_other = chi_xx(x, y, other_x_center, other_y_center, other_radius)

                # 计算其他 chi 的乘积
                product_of_other_chis *= chi(x, y, other_x_center, other_y_center, other_radius)

                # 累加其他 chi 的二阶导数项
                sum_of_other_chi_xx_terms += chi_xx_other

                # 累加交叉项
                sum_of_cross_terms += chi_x_i * chi_x_other

        # 二阶导数乘积法则
        result += chi_xx_i * product_of_other_chis + sum_of_other_chi_xx_terms * chi(x, y, x_center, y_center, radius)
        result += 2 * sum_of_cross_terms

    return result


# multiple_chi_yy: 计算多个 chi 的二阶 y 导数（类似的改法）
def multiple_chi_yy(x, y, circles):
    result = 0
    for i, circle in enumerate(circles):
        x_center, y_center = circle['center']
        radius = circle['radius']

        # 当前 chi 的二阶 y 导数
        chi_yy_i = chi_yy(x, y, x_center, y_center, radius)

        # 当前 chi 的一阶 y 导数
        chi_y_i = chi_y(x, y, x_center, y_center, radius)

        # 计算其他 chi 的乘积
        product_of_other_chis = 1
        sum_of_other_chi_yy_terms = 0
        sum_of_cross_terms = 0

        for j, other_circle in enumerate(circles):
            if i != j:
                other_x_center, other_y_center = other_circle['center']
                other_radius = other_circle['radius']

                # 其他 chi 的一阶和二阶导数
                chi_y_other = chi_y(x, y, other_x_center, other_y_center, other_radius)
                chi_yy_other = chi_yy(x, y, other_x_center, other_y_center, other_radius)

                # 计算其他 chi 的乘积
                product_of_other_chis *= chi(x, y, other_x_center, other_y_center, other_radius)

                # 累加其他 chi 的二阶导数项
                sum_of_other_chi_yy_terms += chi_yy_other

                # 累加交叉项
                sum_of_cross_terms += chi_y_i * chi_y_other

        # 二阶导数乘积法则
        result += chi_yy_i * product_of_other_chis + sum_of_other_chi_yy_terms * chi(x, y, x_center, y_center, radius)
        result += 2 * sum_of_cross_terms

    return result


# 最终拉普拉斯算子
def laplacian_u(x, y, t, circles):
    u_e_value = u_e(x, y, t)
    lap_u_e = laplacian_u_e(x, y, t)
    chi_value = multiple_chi(x, y, circles)

    term1 = 0
    term2_x = 0
    term2_y = 0

    for circle in circles:
        x_center, y_center = circle['center']
        radius = circle['radius']
        term1 += chi_xx(x, y, x_center, y_center, radius) + chi_yy(x, y, x_center, y_center, radius)
        term2_x += chi_x(x, y, x_center, y_center, radius) * (-mu * np.sin(mu * x) * np.sin(nu * y))
        term2_y += chi_y(x, y, x_center, y_center, radius) * (-nu * np.sin(mu * x) * np.sin(nu * y))

    return chi_value * lap_u_e + term1 * u_e_value + 2 * (term2_x + term2_y)


def f_real(x, y, t):
    u_x = mu * np.cos(mu * x) * np.sin(nu * y) * (2 * np.cos(lambda_ * t) + np.sin(lambda_ * t))
    u_xx = -mu ** 2 * np.sin(mu * x) * np.sin(nu * y) * (2 * np.cos(lambda_ * t) + np.sin(lambda_ * t))
    u_y = nu * np.sin(mu * x) * np.cos(nu * y) * (2 * np.cos(lambda_ * t) + np.sin(lambda_ * t))
    u_yy = -nu ** 2 * np.sin(mu * x) * np.sin(nu * y) * (2 * np.cos(lambda_ * t) + np.sin(lambda_ * t))
    u_t = lambda_ * np.sin(mu * x) * np.sin(nu * y) * (np.cos(lambda_ * t) - 2 * np.sin(lambda_ * t))
    return u_t + v_x * u_x + v_y * u_y - D * (u_xx + u_yy)


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

                grad_u_x = []
                grad_u_y = []
                grad_u_t = []
                grad_u_xx = []
                grad_u_yy = []
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

                    grad_u_x.append(u_x.detach().numpy())
                    grad_u_y.append(u_y.detach().numpy())
                    grad_u_t.append(u_t.detach().numpy())
                    grad_u_xx.append(u_xx.detach().numpy())
                    grad_u_yy.append(u_yy.detach().numpy())

                grad_u_x = np.array(grad_u_x).T
                grad_u_y = np.array(grad_u_y).T
                grad_u_t = np.array(grad_u_t).T
                grad_u_xx = np.array(grad_u_xx).T
                grad_u_yy = np.array(grad_u_yy).T

                A_P[:, M_begin: M_begin + M] = grad_u_t + v_x * grad_u_x + v_y * grad_u_y - D * (grad_u_xx + grad_u_yy)
                A_I[:, M_begin: M_begin + M] = ic_values
                A_B[:, M_begin: M_begin + M] = bc_values
                A_O[:, M_begin: M_begin + M] = obs_values

    f_P = f_real(pde_points[:, [0]], pde_points[:, [1]], pde_points[:, [2]])
    f_I = u_e(ic_points[:, [0]], ic_points[:, [1]], ic_points[:, [2]])
    f_B = u_e(bc_points[:, [0]], bc_points[:, [1]], bc_points[:, [2]])
    f_O = u_e(obs_points[:, [0]], obs_points[:, [1]], obs_points[:, [2]])

    c_p = 100.0
    c_i = 1e-5
    c_b = 1e-5
    c_o = 100.0
    # 对每行按其绝对值最大值缩放
    for i in range(len(A_P)):
        ratio = c_p / max(-A_P[i, :].min(), A_P[i, :].max())
        A_P[i, :] = A_P[i, :] * ratio
        f_P[i] = f_P[i] * ratio

    for i in range(len(A_I)):
        ratio = c_i / max(-A_I[i, :].min(), A_I[i, :].max())
        A_I[i, :] = A_I[i, :] * ratio
        f_I[i] = f_I[i] * ratio

    for i in range(len(A_B)):
        ratio = c_b / max(-A_B[i, :].min(), A_B[i, :].max())
        A_B[i, :] = A_B[i, :] * ratio
        f_B[i] = f_B[i] * ratio

    for i in range(len(A_O)):
        ratio = c_o / max(-A_O[i, :].min(), A_O[i, :].max())
        A_O[i, :] = A_O[i, :] * ratio
        f_O[i] = f_O[i] * ratio

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
    ic_points = np.hstack((X[:, :, 0].flatten()[:, None], Y[:, :, 0].flatten()[:, None], T[:, :, 0].flatten()[:, None]))
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

    torch.save(models, 'convection_diffusion_2d_da_complex_domain.pt')
    np.savez('convection_diffusion_2d_da_complex_domain.npz', w=w,
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
    Ms = [100, ]
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
