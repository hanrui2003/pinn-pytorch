import numpy as np
from datetime import datetime
import sys
import matplotlib.pyplot as plt
from matplotlib import animation
from datetime import datetime
import sys
from multiprocessing import Process, current_process, Queue
import concurrent.futures

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


def worker(L, N_interp, x_interp, N_x, N_t, x, dx, dt, x_filter, t_filter):
    sample_num = 100
    u_sample = np.zeros((sample_num, len(t_filter), len(x_filter)))
    for i in range(sample_num):
        y_interp = np.dot(L, np.random.randn(N_interp))
        y_max = y_interp.max()
        y_min = y_interp.min()
        y_interp = 0.1 + 0.1 * (y_interp - y_min) / (y_max - y_min)

        # 检测插值函数
        # fig, ax = plt.subplots()
        # ax.set_xlim(0, 1)
        # ax.set_ylim(0, 0.3)
        # plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
        # line1, = ax.plot(x_interp, y_interp, color="red", linestyle='-', label="t=")
        # plt.show()

        h0 = np.interp(x, x_interp, y_interp)
        m0 = np.zeros_like(h0)
        u0 = np.hstack((h0, m0))

        u = rk4(u0, N_t - 1, dx, dt)
        u_filter = u[t_filter, :][:, x_filter]
        # 把动量转为速度
        u_filter[:, len(x_filter) // 2:] = u_filter[:, len(x_filter) // 2:] / u_filter[:, :len(x_filter) // 2]
        u_sample[i] = u_filter
        print(datetime.now(), 'pid-', current_process().pid, str(i + 1) + "th sample done.")

    return u_sample


if "__main__" == __name__:
    # np.random.seed(123)
    # 插值点个数
    N_interp = 1001
    # 扰动项，防止后续的 cholesky 分解过程中，有些元素非常小甚至为零时，导致平方根运算不稳定。
    jitter = 1e-10
    x_interp = np.linspace(0, 1, N_interp)
    l = 0.2
    K = np.exp(-0.5 * (x_interp[:, None] - x_interp) ** 2 / (l ** 2))
    # cholesky 分解，得到下三角矩阵
    L = np.linalg.cholesky(K + jitter * np.eye(N_interp))

    # 数值解配置
    N_x = 401
    N_t = 2401
    print("N_x :", N_x, "N_t :", N_t)

    x = np.linspace(0, 1, N_x)
    t = np.linspace(0, 6, N_t)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    print('dx:', dx, 'dt:', dt)

    x_filter1 = [i for i in range(0, N_x, 4)]
    x_filter2 = [i + N_x for i in range(0, N_x, 4)]
    x_filter = x_filter1 + x_filter2
    t_filter = [i for i in range(0, N_t, 4)]

    # 记录训练开始时间
    start_time = datetime.now()
    print("Sampling started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))

    # 进程数
    p_num = 10

    # 创建 ProcessPoolExecutor
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # 提交任务给执行器
        futures = [executor.submit(worker, L, N_interp, x_interp, N_x, N_t, x, dx, dt, x_filter, t_filter) for _ in
                   range(p_num)]

        # 获取任务的结果
        u_sample_all = np.empty((0, len(t_filter), len(x_filter)))
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            u_sample_all = np.concatenate((u_sample_all, result), axis=0)

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Sampling ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)

    np.save(r"swe_1d_rbf_sample_l=0.2.npy", u_sample_all)
    print(datetime.now(), 'sample size(GB) :', sys.getsizeof(u_sample_all) / 1024 / 1024 / 1024)
