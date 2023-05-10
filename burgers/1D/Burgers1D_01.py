import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


def RBF(x1, x2, output_scale=1.0, length_scale=0.2):
    """
    高斯径向基函数，用于生成协方差矩阵
    :param x1:随机向量
    :param x2:随机向量
    """
    diffs = np.expand_dims(x1 / length_scale, 1) - np.expand_dims(x2 / length_scale, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


def gp_sample(num=5000):
    """
    高斯过程采样，生成随机函数，函数定义域[0,1],对于定义域其实其他的，只要缩放即可；
    :param num:需要生成的函数个数
    :return:随机生成的函数列表
    """
    # 随机采样的向量维度，也是插值函数的点数
    N = 500
    x_lb = 0.
    x_ub = 1.
    # 扰动项，防止后续的 cholesky 分解过程中，有些元素非常小甚至为零时，导致平方根运算不稳定。
    jitter = 1e-10
    x = np.linspace(x_lb, x_ub, N)[:, None]
    # 经过核函数运算得到的协方差矩阵
    K = RBF(x, x)
    # cholesky 分解，得到下三角矩阵
    L = np.linalg.cholesky(K + jitter * np.eye(N))

    func_list = []
    for _ in range(num):
        # 生成N维标准正态分布的向量，与L内积，得到符合协方差为K的正态分布向量
        y_sample = np.dot(L, np.random.randn(N))
        # 分段线性差值
        func_list.append(lambda x0, y=y_sample: np.interp(x0, x.flatten(), y))

    return func_list


def generate_obs(num=5000):
    """
    选取一些观测点，并对这些点生成一批观测数据。
    :param num:观测数据的数量
    :return:
    """
    ic_func_list = gp_sample(num)

    # 以下是数值解
    Nx = 50
    Nt = 100
    D = 0.01
    k = 0.01

    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2
    print("grid ratio", D * dt / h2)

    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    # numpy的meshgrid参数，第一个是横轴，正方向向右，第二个是纵轴，正方向向下，坐标原点左上角；
    X, T = np.meshgrid(x, t)

    # 选取观测点的子网格，就是对大网格取子矩阵
    # x和t每隔多少个点取一个点
    x_interval = Nx // 10
    t_interval = Nt // 10
    idx_x = [i for i in range(x_interval, Nx, x_interval)]
    idx_t = [i for i in range(t_interval, Nt, t_interval)]

    x_train = X[idx_t][:, idx_x].flatten()[:, None]
    t_train = T[idx_t][:, idx_x].flatten()[:, None]
    y_train = np.hstack((x_train, t_train))

    # 观测训练数据
    o_train = []
    for func in ic_func_list:
        U = np.zeros((Nt, Nx))
        U[0] = func(x)
        for n in range(Nt - 1):
            u_n = U[n]
            u_next = u_n + dt * ((D / h2) * D2 @ u_n + k * u_n ** 2)
            u_next[np.array([0, -1])] = 0.
            U[n + 1] = u_next

        o_train.append(U[idx_t][:, idx_x].flatten())

    return np.asarray(o_train), y_train


if "__main__" == __name__:
    torch.manual_seed(123)
    np.random.seed(123)

    generate_obs(1)
