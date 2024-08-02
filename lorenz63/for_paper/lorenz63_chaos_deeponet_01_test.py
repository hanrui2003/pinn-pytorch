import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
import torch.autograd as autograd
from lorenz63_chaos_deeponet_01 import L63Net
from lorenz63_data_chaos import u_next
import matplotlib.pyplot as plt

if "__main__" == __name__:
    np.random.seed(123)

    # 随机生成初值点，
    # 先根据数值解的结果，使用核密度估计，然后再采样
    U = np.load('lorenz63_chaos.npy')
    # 定义带宽范围
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    # 网格搜索最优带宽
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=5)
    grid.fit(U)
    best_bandwidth = grid.best_params_['bandwidth']
    print("best bandwidth", best_bandwidth)
    # 拟合KDE模型
    kde = KernelDensity(kernel='gaussian', bandwidth=best_bandwidth).fit(U)
    u0 = kde.sample(1)
    # u0 = U[0]
    print("u0", u0)

    # 步长
    h = 0.005
    total_points = 1001
    t = np.linspace(0, 5, total_points)

    u_truth = np.zeros([total_points, 3])
    u_truth[0] = u0
    for j in range(1, total_points):
        u_truth[j] = u_next(u_truth[j - 1], h)

    N_delta_t = 21
    delta_t = np.linspace(0, 0.1, N_delta_t)
    t_test = delta_t[:, None]

    u0_test = np.tile(u0, (N_delta_t, 1))
    s_test = np.hstack((t_test, u0_test))
    s_test = torch.from_numpy(s_test).float()

    model = torch.load('lorenz63_chaos_deeponet_01_0.005.pt', map_location=torch.device('cpu'))

    u_pred = np.zeros((total_points, 3))
    for i in range(50):
        u_hat = model(s_test).detach().numpy()
        u_pred[i * 20:(i + 1) * 20] = u_hat[0:20]
        # u0_test = np.tile(u_hat[-1], (N_delta_t, 1))
        # 改为引入观测，用数值解对应的值作为观测值
        if i + 1 == 50:
            break
        u0_test = np.tile(u_truth[(i + 1) * 20], (N_delta_t, 1))
        s_test = np.hstack((t_test, u0_test))
        s_test = torch.from_numpy(s_test).float()

    # 去除最后段端点
    t = t[:-1]
    u = u_truth[:-1]
    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]
    u_hat = u_pred[:-1]
    x_hat = u_hat[:, 0]
    y_hat = u_hat[:, 1]
    z_hat = u_hat[:, 2]

    epsilon = np.abs(u - u_hat)
    L_inf_error = np.max(epsilon)
    L_2_error = np.sqrt(np.sum(epsilon ** 2) / epsilon.size)
    L_rel_error = np.linalg.norm(epsilon) / np.linalg.norm(u)
    print('L_inf_error :', L_inf_error, ' , L_2_error', L_2_error, ' , L_rel_error : ', L_rel_error)

    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.plot(t, x, color='b', label='x')
    ax1.plot(t, y, color='g', label='y')
    ax1.plot(t, z, color='k', label='z')
    ax1.plot(t, x_hat, 'r--', label='x_hat')
    ax1.plot(t, y_hat, 'c--', label='y_hat')
    ax1.plot(t, z_hat, color='orange', linestyle='--', label='z_hat')
    ax1.set_xlabel('t', color='black')
    ax1.set_ylabel('u(t)', color='black')
    ax1.legend(loc='upper right')

    ax2.plot(x, y, z, 'r', label='RK')
    ax2.plot(x_hat, y_hat, z_hat, color='b', linestyle='--', label='DeepONet')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    plt.show()
