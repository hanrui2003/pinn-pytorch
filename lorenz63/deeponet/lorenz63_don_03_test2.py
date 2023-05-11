import numpy as np
import torch
from lorenz63_don_03 import L63Net, SinActivation
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from lorenz63_data_chaos import u_next

"""
训练小区间长度为0.1的测试类
"""


def plot(t, u_truth, u_pred):
    """
    同时绘制数值解和神经网络解，上面数值解，下面神经网络解。
    """
    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')

    ax1.plot(t, u_truth[:, 0], color='r', label='x')
    ax1.plot(t, u_truth[:, 1], color='g', label='y')
    ax1.plot(t, u_truth[:, 2], color='b', label='z')

    ax2.plot(u_truth[:, 0], u_truth[:, 1], u_truth[:, 2], color='r')
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')
    ax2.set_title('RK')

    ax3.plot(t, u_pred[:, 0], color='r', label='x')
    ax3.plot(t, u_pred[:, 1], color='g', label='y')
    ax3.plot(t, u_pred[:, 2], color='b', label='z')

    ax4.plot(u_pred[:, 0], u_pred[:, 1], u_pred[:, 2], color='r')
    ax4.set_xlabel('x')
    ax4.set_ylabel('y')
    ax4.set_zlabel('z')
    ax4.set_title('DeepONet')

    plt.show()


if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)

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
    print("u0", u0)

    # 总点数
    total_points = 1000
    # 步长
    h = 0.005
    t_total = np.arange(0., total_points * h, h)

    # 以下是数值解
    # 计算三千个个点
    u_truth = np.zeros([total_points, 3])
    u_truth[0] = u0

    for j in range(1, total_points):
        u_truth[j] = u_next(u_truth[j - 1], h)

    # 以下是神经网络解
    # 神经网络训练时，区间的上下界
    t_lb = 0.
    t_ub = 0.1
    # 这里让能取到端点，所以在t_ub加上一个小量
    t = np.arange(t_lb, t_ub + h / 2, h)
    t_test = torch.from_numpy(t[:, None]).float()
    N_t = len(t_test)

    u0_test = np.tile(u0, (N_t, 1))
    u0_test = torch.from_numpy(u0_test).float()

    model = torch.load('lorenz63_don_03_0.1_0.01_gzz_01.pt', map_location=torch.device('cpu'))

    u_pred = np.zeros((total_points, 3))
    for i in range(50):
        u_hat = model(u0_test, t_test).detach().numpy()
        u_pred[i * 20:(i + 1) * 20] = u_hat[0:20]
        u0_test[:] = torch.from_numpy(u_hat[-1])

    max_error = abs(u_truth - u_pred).max(axis=0)
    print("max_error", max_error)
    # 计算相对误差，(真值-预测值)的范数/真值的范数
    error = np.linalg.norm(u_truth - u_pred) / np.linalg.norm(u_truth)
    print("error", error)
    plot(t_total, u_truth, u_pred)
