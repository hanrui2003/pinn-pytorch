import torch
import torch.nn as nn
import numpy as np
import time
import math
from scipy.linalg import lstsq, block_diag
import matplotlib.pyplot as plt
from datetime import datetime

# computational domain，定义域的左右边界
X_min = -1.0
X_max = 1.0
T_min = 0.0
T_max = 1.0


# random initialization for parameters
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a=-R_m, b=R_m)
        nn.init.uniform_(m.bias, a=-R_m, b=R_m)


# 类似于自定义激活函数，定义基函数phi
# random feature basis when using \psi^{a} as PoU function
class RFM_rep_a(nn.Module):
    def __init__(self, d, J_n, x_min, x_max, t_min, t_max):
        """
        创建一个单隐层的全连接网络
        d:输入维度
        J_n:隐层维度，注意这里也是输出维度，因为后续没有聚合层。因为内层的参数固定，聚合层交给后续的外围参数优化。
        x_min: 单位分解区间的左端
        x_max: 单位分解区间的右端
        """
        super(RFM_rep_a, self).__init__()
        self.d = d
        self.J_n = J_n
        # 小区间半径，区间长度的一半
        self.r = torch.tensor(((x_max - x_min) / 2.0, (t_max - t_min) / 2.0))
        # 小区间中心
        self.x_c = torch.tensor(((x_max + x_min) / 2, (t_max + t_min) / 2))
        # 这里就是个简单的线性层+Tanh，注意这里是phi，不是psi
        self.phi = nn.Sequential(nn.Linear(self.d, self.J_n, bias=True), nn.Tanh())

    def forward(self, x):
        # 标准化，使得取值在[-1,1]
        x = (x - self.x_c) / self.r
        x = self.phi(x)
        return x


# random feature basis when using \psi^{b} as PoU function
class RFM_rep_b(nn.Module):
    def __init__(self, d, J_n, x_max, x_min):
        super(RFM_rep_b, self).__init__()
        self.d = d
        self.J_n = J_n
        self.n = x_min / (X_max - X_min) * M_p
        self.r = (x_max - x_min) / 2.0
        self.x_0 = (x_max + x_min) / 2
        self.phi = nn.Sequential(nn.Linear(self.d, self.J_n, bias=True), nn.Tanh())

    def forward(self, x):
        d = (x - self.x_0) / self.r
        psi = ((d <= -3 / 4) & (d > -5 / 4)) * (1 + torch.sin(2 * np.pi * d)) / 2 + (
                (d <= 3 / 4) & (d > -3 / 4)) * 1.0 + ((d <= 5 / 4) & (d > 3 / 4)) * (
                      1 - torch.sin(2 * np.pi * d)) / 2
        y = self.phi(d)
        if self.n == 0:
            psi = ((d <= 3 / 4) & (d > -1)) * 1.0 + ((d <= 5 / 4) & (d > 3 / 4)) * (1 - torch.sin(2 * np.pi * d)) / 2
        elif self.n == M_p - 1:
            psi = ((d <= -3 / 4) & (d > -5 / 4)) * (1 + torch.sin(2 * np.pi * d)) / 2 + ((d <= 1) & (d > -3 / 4)) * 1.0
        else:
            psi = ((d <= -3 / 4) & (d > -5 / 4)) * (1 + torch.sin(2 * np.pi * d)) / 2 + (
                    (d <= 3 / 4) & (d > -3 / 4)) * 1.0 + ((d <= 5 / 4) & (d > 3 / 4)) * (
                          1 - torch.sin(2 * np.pi * d)) / 2
        return psi * y


# predefine the random feature functions in each PoU region
def pre_define(M_p, J_n):
    """
    为每个单位分解子区间生成一个神经网络
    M_p：单位分解区间的数量(M_p[0]*M_p[1])，对应着单隐层神经网络的数量
    J_n：神经网络隐层的维度
    """
    models = []
    # 空间维度(x)
    for i in range(M_p[0]):
        model_x = []
        # 单位分解子区间的左端(空间维度)
        x_min = (X_max - X_min) / M_p[0] * i + X_min
        # 单位分解子区间的右端(空间维度)
        x_max = (X_max - X_min) / M_p[0] * (i + 1) + X_min
        # 时间维度(t)
        for j in range(M_p[1]):
            # 单位分解子区间的左端(时间维度)
            t_min = (T_max - T_min) / M_p[1] * j + T_min
            # 单位分解子区间的右端(时间维度)
            t_max = (T_max - T_min) / M_p[1] * (j + 1) + T_min
            # 每个单位分解区间对应一个FCN
            model = RFM_rep_a(d=2, J_n=J_n, x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max)
            model = model.apply(weights_init)
            # 将模型中所有的浮点数参数（权重、偏置等）转换为 double，就是把默认的float32转成float64
            model = model.double()
            # 内层参数固定，不需要反向传播更新，所以去除梯度跟踪
            for param in model.parameters():
                param.requires_grad = False

            model_x.append(model)

        models.append(model_x)
    return models


# Assembling the matrix A,f in linear system 'Au=f'
def assemble_matrix(models, points, M_p, J_n, Q):
    """
    models：模型
    points：配点列表
    M_p:划分的区间数,M_p[0]*M_p[1]
    Q：每个小区间取配点的个数(每个维度)，实际为Q+1
    J_n：线性层的输出维度
    """
    # 所有PDE条件
    A_P = np.zeros([M_p[0] * M_p[1] * Q * Q, M_p[0] * M_p[1] * J_n])
    # 初值条件
    A_I = np.zeros([M_p[0] * (Q + 1), M_p[0] * M_p[1] * J_n])
    # 边界条件
    A_B = np.zeros([2 * M_p[1] * (Q + 1), M_p[0] * M_p[1] * J_n])
    # 0阶连续性条件
    A_C0_up_down = np.zeros([M_p[0] * (M_p[1] - 1) * (Q + 1), M_p[0] * M_p[1] * J_n])
    A_C0_left_right = np.zeros([M_p[1] * (M_p[0] - 1) * (Q + 1), M_p[0] * M_p[1] * J_n])
    # 1阶连续性条件
    A_C1_up_down = np.zeros([M_p[0] * (M_p[1] - 1) * (Q + 1), M_p[0] * M_p[1] * J_n])
    A_C1_left_right = np.zeros([M_p[1] * (M_p[0] - 1) * (Q + 1), M_p[0] * M_p[1] * J_n])

    # 右端项
    f_A_P = np.zeros((len(A_P), 1))
    f_A_I = np.zeros((len(A_I), 1))
    f_A_B = np.zeros((len(A_B), 1))
    f_A_C0_up_down = np.zeros((len(A_C0_up_down), 1))
    f_A_C0_left_right = np.zeros((len(A_C0_left_right), 1))
    f_A_C1_up_down = np.zeros((len(A_C1_up_down), 1))
    f_A_C1_left_right = np.zeros((len(A_C1_left_right), 1))

    # 去掉右边界和上边界后剩的点的索引，后面用到，这些点是用来算PDE的
    pde_idx_filter = [m * (Q + 1) + n for m in range(Q) for n in range(Q)]

    # 边值条件的计数，后面用到
    bc_idx_filter1 = [m * (Q + 1) for m in range(Q + 1)]
    bc_idx_filter2 = [(m + 1) * (Q + 1) - 1 for m in range(Q + 1)]
    bc_count = 0

    # 光滑性条件的索引，后面用到
    smooth_idx_filter1 = [m * (Q + 1) for m in range(Q + 1)]
    smooth_idx_filter2 = [(m + 1) * (Q + 1) - 1 for m in range(Q + 1)]

    # 遍历每个区间，区间数M_p =========== start
    for i in range(M_p[0]):
        for j in range(M_p[1]):
            # 当前区间的配点
            point = torch.tensor(points[i][j], requires_grad=True)
            # 当前单位分解区间的配点进入对应的神经网络
            out = models[i][j](point)
            values = out.detach().numpy()

            # 记录一阶导
            grad_u_x = []  # 这个用于一阶光滑性条件
            grad_u_t = []
            grad_u_xx = []
            # 当前区间的配点求导，由于torch的局限，不能多维函数（J_n维）一起求导，所以只能循环
            for k in range(J_n):
                u_x_t = torch.autograd.grad(outputs=out[:, k], inputs=point,
                                            grad_outputs=torch.ones_like(out[:, k]),
                                            create_graph=True, retain_graph=True)[0]

                u_xx_tt = torch.autograd.grad(outputs=u_x_t, inputs=point,
                                              grad_outputs=torch.ones_like(u_x_t),
                                              create_graph=True, retain_graph=True)[0]

                u_x = u_x_t[:, 0]
                u_t = u_x_t[:, 1]
                u_xx = u_xx_tt[:, 0]

                grad_u_x.append(u_x.detach().numpy())
                grad_u_t.append(u_t.detach().numpy())
                grad_u_xx.append(u_xx.detach().numpy())

            grad_u_x = np.array(grad_u_x).T
            grad_u_t = np.array(grad_u_t).T
            grad_u_xx = np.array(grad_u_xx).T

            # 类似微分算子，注意这里不等同于微分算子，因为这里的Lu对于一个单点x，其输出维度是J_n
            Lu = grad_u_xx - grad_u_t

            # Lu = f condition
            # 构造分块对角矩阵，每个块Q行，J_n列，每个块就是一个小区间（扣除区间右端点）对应的方阵
            # i * M_p[1] + j 表示当前是第几个区间
            # Q * Q 该区间中用于计算的条件数，Q * Q 是配点数
            A_P[(i * M_p[1] + j) * Q * Q:(i * M_p[1] + j + 1) * Q * Q,
            (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = Lu[pde_idx_filter]
            f_A_P[(i * M_p[1] + j) * Q * Q:(i * M_p[1] + j + 1) * Q * Q, :] = (np.exp(-points[i][j][:, [1]]) * (
                    1 - np.pi ** 2) * np.sin(np.pi * points[i][j][:, [0]]))[pde_idx_filter]

            # 初值条件
            if j == 0:
                A_I[i * (Q + 1):(i + 1) * (Q + 1), (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = values[:Q + 1]
                f_A_I[i * (Q + 1):(i + 1) * (Q + 1), :] = np.sin(np.pi * points[i][j][:, [0]])[:Q + 1]

            # 边界条件
            # 注意边界的输出不需要经过微分算子运算，因为是比对标签值
            if i == 0:
                A_B[bc_count * (Q + 1):(bc_count + 1) * (Q + 1), (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] \
                    = values[bc_idx_filter1]
                bc_count += 1

            if i == M_p[0] - 1:
                A_B[bc_count * (Q + 1):(bc_count + 1) * (Q + 1), (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] \
                    = values[bc_idx_filter2]
                bc_count += 1

            # smoothness conditions
            # 光滑性处理，当划分为多个区间时，区间连接的地方做光滑处理：
            # 看上下接触点
            if j == 0:
                # 当前子区间的上面，取负值
                A_C0_up_down[(i * (M_p[1] - 1) + j) * (Q + 1):(i * (M_p[1] - 1) + j + 1) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = -values[-(Q + 1):]
                # 当前子区间的上面，取负值
                A_C1_up_down[(i * (M_p[1] - 1) + j) * (Q + 1):(i * (M_p[1] - 1) + j + 1) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = -grad_u_x[-(Q + 1):]
            elif j == M_p[1] - 1:
                # 当前子区间的下面，取正值
                A_C0_up_down[(i * (M_p[1] - 1) + j - 1) * (Q + 1):(i * (M_p[1] - 1) + j) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = values[:Q + 1]
                # 当前子区间的下面，取正值
                A_C1_up_down[(i * (M_p[1] - 1) + j - 1) * (Q + 1):(i * (M_p[1] - 1) + j) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = grad_u_x[:Q + 1]
            else:
                # 当前子区间的上面，取负值
                A_C0_up_down[(i * (M_p[1] - 1) + j) * (Q + 1):(i * (M_p[1] - 1) + j + 1) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = -values[-(Q + 1):]
                # 当前子区间的下面，取正值
                A_C0_up_down[(i * (M_p[1] - 1) + j - 1) * (Q + 1):(i * (M_p[1] - 1) + j) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = values[:Q + 1]
                # 当前子区间的上面，取负值
                A_C1_up_down[(i * (M_p[1] - 1) + j) * (Q + 1):(i * (M_p[1] - 1) + j + 1) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = -grad_u_x[-(Q + 1):]
                # 当前子区间的下面，取正值
                A_C1_up_down[(i * (M_p[1] - 1) + j - 1) * (Q + 1):(i * (M_p[1] - 1) + j) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = grad_u_x[:Q + 1]

            # 看左右接触点
            if i == 0:
                # 当前子区间的右面，取负值
                A_C0_left_right[(j * (M_p[0] - 1) + i) * (Q + 1):(j * (M_p[0] - 1) + i + 1) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = -values[smooth_idx_filter2]
                # 当前子区间的右面，取负值
                A_C1_left_right[(j * (M_p[0] - 1) + i) * (Q + 1):(j * (M_p[0] - 1) + i + 1) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = -grad_u_x[smooth_idx_filter2]
            elif i == M_p[0] - 1:
                # 当前子区间的左面，取正值
                A_C0_left_right[(j * (M_p[0] - 1) + i - 1) * (Q + 1):(j * (M_p[0] - 1) + i) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = values[smooth_idx_filter1]
                # 当前子区间的左面，取正值
                A_C1_left_right[(j * (M_p[0] - 1) + i - 1) * (Q + 1):(j * (M_p[0] - 1) + i) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = grad_u_x[smooth_idx_filter1]
            else:
                # 当前子区间的右面，取负值
                A_C0_left_right[(j * (M_p[0] - 1) + i) * (Q + 1):(j * (M_p[0] - 1) + i + 1) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = -values[smooth_idx_filter2]
                # 当前子区间的左面，取正值
                A_C0_left_right[(j * (M_p[0] - 1) + i - 1) * (Q + 1):(j * (M_p[0] - 1) + i) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = values[smooth_idx_filter1]
                # 当前子区间的右面，取负值
                A_C1_left_right[(j * (M_p[0] - 1) + i) * (Q + 1):(j * (M_p[0] - 1) + i + 1) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = -grad_u_x[smooth_idx_filter2]
                # 当前子区间的左面，取正值
                A_C1_left_right[(j * (M_p[0] - 1) + i - 1) * (Q + 1):(j * (M_p[0] - 1) + i) * (Q + 1),
                (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = grad_u_x[smooth_idx_filter1]

    # 遍历每个区间 =========== end
    print("assemble A and f")
    A = np.concatenate((A_P, A_I, A_B, A_C0_up_down, A_C0_left_right, A_C1_up_down, A_C1_left_right), axis=0)
    f = np.concatenate((f_A_P, f_A_I, f_A_B, f_A_C0_up_down, f_A_C0_left_right, f_A_C1_up_down, f_A_C1_left_right),
                       axis=0)

    return A, f


def main(M_p, J_n, Q):
    # prepare collocation points
    time_begin = time.time()
    points = []
    # 把定义域划分成多个子区间
    # 空间维度（x）
    for i in range(M_p[0]):
        point_x = []
        # 单位分解子区间的左端(空间维度)
        x_min = (X_max - X_min) / M_p[0] * i + X_min
        # 单位分解子区间的右端(空间维度)
        x_max = (X_max - X_min) / M_p[0] * (i + 1) + X_min
        # 时间维度(t)
        for j in range(M_p[1]):
            # 单位分解子区间的左端(时间维度)
            t_min = (T_max - T_min) / M_p[1] * j + T_min
            # 单位分解子区间的右端(时间维度)
            t_max = (T_max - T_min) / M_p[1] * (j + 1) + T_min
            # 单位分解子区间的配点，Q表示每个单位分解子区间划分的子区间，所以算上端点，配点数为Q+1
            x = np.linspace(x_min, x_max, Q + 1)
            t = np.linspace(t_min, t_max, Q + 1)
            X, T = np.meshgrid(x, t)
            point_x.append(np.hstack((X.flatten()[:, None], T.flatten()[:, None])))
        points.append(point_x)

    # prepare models
    # 一个单位分解子区间对应一个单隐层的全连接网络
    models = pre_define(M_p, J_n)

    # matrix define (Au=f)
    A, f = assemble_matrix(models, points, M_p, J_n, Q)
    print('***********************')
    print('A shape: ', A.shape, 'f shape: ', f.shape)
    # rescaling
    # 这个缩放因子，对于其他方程，该如何确定？？？
    c = 100.0
    # 对每行按其绝对值最大值缩放
    # 为什么不按照绝对值的最大值缩放？这样会映射到[-c,c]，岂不完美？看文档应该是绝对值，代码错误？还有就是最大值有极小的概率是0，也是个风险点。
    for i in range(len(A)):
        ratio = c / max(-A[i, :].min(), A[i, :].max())
        A[i, :] = A[i, :] * ratio
        f[i] = f[i] * ratio

    # solve
    # 求 Aw=f的最小二乘解，这里w就是外围参数，可以近似认为是神经网络的隐层到输出层的权重参数。
    print(datetime.now(), "start least square")
    w = lstsq(A, f)[0]

    # test
    print(datetime.now(), "test start :", time.time())
    error = test(models, M_p, J_n, Q, w)
    print(datetime.now(), "error :", error)
    print(datetime.now(), "test end :", time.time())
    return error


def f_real(x, t):
    return np.exp(-t) * np.sin(np.pi * x)


def plot(X1, T1, U1, X2, T2, U2):
    """
    同时绘制PDE的数值解和神经网络解，上面数值解，下面神经网络解。
    """
    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(221)
    ax2 = fig.add_subplot(222, projection='3d')
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224, projection='3d')

    cp1 = ax1.contourf(T1, X1, U1, 20, cmap="rainbow")
    fig.colorbar(cp1, ax=ax1)
    ax1.set_title('u(x,t)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')

    ax2.plot_surface(T1, X1, U1, cmap="rainbow")
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u(x,t)')

    cp3 = ax3.contourf(T2, X2, U2, 20, cmap="rainbow")
    fig.colorbar(cp3, ax=ax3)
    ax3.set_title('NN(x,t)')
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')

    ax4.plot_surface(T2, X2, U2, cmap="rainbow")
    ax4.set_xlabel('t')
    ax4.set_ylabel('x')
    ax4.set_zlabel('RFM(x,t)')

    plt.show()


# calculate the l^{infinity}-norm and l^{2}-norm error for u
def test(models, M_p, J_n, Q, w):
    # 测试的时候，把网格变细，网格大小为配点的一半3
    test_Q = 2 * Q
    U_numerical = np.zeros((M_p[1] * test_Q + 1, M_p[0] * test_Q + 1))
    for i in range(M_p[0]):
        # 单位分解子区间的左端(空间维度)
        x_min = (X_max - X_min) / M_p[0] * i + X_min
        # 单位分解子区间的右端(空间维度)
        x_max = (X_max - X_min) / M_p[0] * (i + 1) + X_min
        for j in range(M_p[1]):
            # 单位分解子区间的左端(时间维度)
            t_min = (T_max - T_min) / M_p[1] * j + T_min
            # 单位分解子区间的右端(时间维度)
            t_max = (T_max - T_min) / M_p[1] * (j + 1) + T_min
            # 单位分解子区间的配点，Q表示每个单位分解子区间划分的子区间，所以算上端点，配点数为Q+1
            x = np.linspace(x_min, x_max, test_Q + 1)
            t = np.linspace(t_min, t_max, test_Q + 1)
            X, T = np.meshgrid(x, t)
            point = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
            point = torch.tensor(point, requires_grad=False)
            out = models[i][j](point)
            values = out.detach().numpy()
            numerical_value = np.dot(values, w[(i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n]).reshape(-1,
                                                                                                           test_Q + 1)
            U_numerical[j * test_Q:(j + 1) * test_Q + 1, i * test_Q:(i + 1) * test_Q + 1] = numerical_value

    x = np.linspace(X_min, X_max, M_p[0] * test_Q + 1)
    t = np.linspace(T_min, T_max, M_p[1] * test_Q + 1)
    X, T = np.meshgrid(x, t)
    U_true = f_real(X, T)
    error = np.linalg.norm(U_true - U_numerical) / np.linalg.norm(U_true)
    print(datetime.now(), "relative error: ", error)
    plot(X, T, U_true, X, T, U_numerical)

    return error


if __name__ == '__main__':
    # 固定网络初始化参数，用于debug
    # torch.manual_seed(123)
    # 超参数：随机特征（weight+bias）的均匀分布范围
    R_m = 2
    # 超参数：每个区间随机特征函数的个数，即每个区间对应的神经网络的隐层的维度
    J_n = 50  # the number of basis functions per PoU region
    # 超参数：每个区域配点的个数，其实配点个数是Q+1,这里的Q是每个单位分解区间的等分的区间数，注意这里针对的是每个维度
    Q = 50  # the number of collocation points per PoU region
    # 超参数：单位分解的区间数，注意这里指的是每个维度都划分为M_p个区间
    M_p = 4, 2
    main(M_p, J_n, Q)
