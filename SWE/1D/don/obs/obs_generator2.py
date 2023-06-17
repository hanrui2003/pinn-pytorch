import numpy as np
from datetime import datetime
import sys

"""
数据区间[0,1]*[0,0.5]
"""


def dudt(u_n, dx):
    # 波速
    c = 0.2
    # 重力加速度
    g = 1.0
    length = len(u_n)
    half_len = length // 2
    curr_dhdt = np.zeros(half_len)
    curr_dmdt = np.zeros(half_len)
    curr_h = u_n[:half_len]
    curr_m = u_n[half_len:]
    for i in range(half_len):
        if i == 0:
            curr_dhdt[i] = (-curr_m[i] - curr_m[i + 1] + c * (curr_h[i] - 2 * curr_h[i] + curr_h[i + 1])) / (2 * dx)
            curr_dmdt[i] = ((curr_m[i] ** 2 / curr_h[i] - curr_m[i + 1] ** 2 / curr_h[i + 1]) + (
                    curr_h[i] ** 2 - curr_h[i + 1] ** 2) * g / 2 + c * (
                                    -curr_m[i] - 2 * curr_m[i] + curr_m[i + 1])) / (2 * dx)
        elif i == half_len - 1:
            curr_dhdt[i] = (curr_m[i - 1] + curr_m[i] + c * (curr_h[i - 1] - 2 * curr_h[i] + curr_h[i])) / (2 * dx)
            curr_dmdt[i] = ((curr_m[i - 1] ** 2 / curr_h[i - 1] - curr_m[i] ** 2 / curr_h[i]) + (
                    curr_h[i - 1] ** 2 - curr_h[i] ** 2) * g / 2 + c * (
                                    curr_m[i - 1] - 2 * curr_m[i] - curr_m[i])) / (2 * dx)
        else:
            curr_dhdt[i] = (curr_m[i - 1] - curr_m[i + 1] + c * (curr_h[i - 1] - 2 * curr_h[i] + curr_h[i + 1])) / (
                    2 * dx)
            curr_dmdt[i] = ((curr_m[i - 1] ** 2 / curr_h[i - 1] - curr_m[i + 1] ** 2 / curr_h[i + 1]) + (
                    curr_h[i - 1] ** 2 - curr_h[i + 1] ** 2) * g / 2 + c * (
                                    curr_m[i - 1] - 2 * curr_m[i] + curr_m[i + 1])) / (2 * dx)

    return np.hstack((curr_dhdt, curr_dmdt))


def rk4(u0, n, dx, dt):
    """

    :param u0: 初值
    :param n: 迭代次数
    :param dt: 时间步长
    :return: 数值解
    """
    u = np.zeros((n + 1, len(u0)))
    u[0] = u0
    for i in range(n):
        k1 = dudt(u[i], dx)
        k2 = dudt(u[i] + 0.5 * dt * k1, dx)
        k3 = dudt(u[i] + 0.5 * dt * k2, dx)
        k4 = dudt(u[i] + dt * k3, dx)
        u[i + 1] = u[i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return u


if "__main__" == __name__:
    N_x = 401
    N_t = 201
    print("N_x :", N_x, "N_t :", N_t)

    x_step = (N_x - 1) // 10
    t_step = (N_t - 1) // 5
    print("x_step :", x_step, "t_step :", t_step)

    idx_x = [i for i in range(0, N_x, x_step)]
    x_count = len(idx_x)
    idx_x2 = idx_x + [i + N_x for i in range(0, N_x, x_step)]
    idx_t = [i for i in range(0, N_t, t_step)]
    t_count = len(idx_t)
    print("idx_x2 :", idx_x2, "idx_t :", idx_t)

    x = np.linspace(0, 1, N_x)
    t = np.linspace(0, 0.5, N_t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]

    x_train_point = x[idx_x]
    t_train_point = t[idx_t]
    X, T = np.meshgrid(x_train_point, t_train_point)
    x_train = X.flatten()[:, None]
    t_train = T.flatten()[:, None]
    y_train = np.hstack((x_train, t_train))

    # mu 控制着样本数量
    mu = np.linspace(0, 1, 101)
    h0_list = 0.1 + 0.1 * np.exp(-64 * (x - mu[:, None]) ** 2)
    m0_list = np.zeros_like(h0_list)
    u0_list = np.hstack((h0_list, m0_list))

    # 记录训练开始时间
    start_time = datetime.now()
    print("Process started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    # 观测训练数据
    o_train = []
    for index, u0 in enumerate(u0_list):
        u = rk4(u0, N_t - 1, dx, dt)
        tmp = u[idx_t, :][:, idx_x2]
        h = tmp[:, :x_count].flatten()
        m = tmp[:, x_count:].flatten()
        v = m / h
        o_train.append(np.hstack((h, v)))
        print(datetime.now(), 'generated obs :', index + 1)

    np.savez('swe_1d_don_obs_05.npz', o_train=np.asarray(o_train), y_train=y_train)

    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Process ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
    print("o_train size:", sys.getsizeof(o_train) / 1024 / 1024)
