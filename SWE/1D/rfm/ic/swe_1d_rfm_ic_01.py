import torch
import torch.nn as nn
import numpy as np
import time
import math
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

# computational domain，定义域的左右边界
X_min = 0.0
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
    M_p：单位分解区间的数量(M_p*M_p)，对应着单隐层神经网络的数量
    J_n：神经网络隐层的维度
    """
    models = []
    # 空间维度(x)
    for i in range(M_p):
        model_x = []
        # 时间维度(t)
        for j in range(M_p):
            # 单位分解子区间的左端(空间维度)
            x_min = (X_max - X_min) / M_p * i + X_min
            # 单位分解子区间的右端(空间维度)
            x_max = (X_max - X_min) / M_p * (i + 1) + X_min
            # 单位分解子区间的左端(时间维度)
            t_min = (T_max - T_min) / M_p * j + T_min
            # 单位分解子区间的右端(时间维度)
            t_max = (T_max - T_min) / M_p * (j + 1) + T_min
            # 每个单位分解区间对应两个FCN（one for h, one for v）
            model_x_t = []
            model_for_h = RFM_rep_a(d=2, J_n=J_n, x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max)
            model_for_h = model_for_h.apply(weights_init)
            # 将模型中所有的浮点数参数（权重、偏置等）转换为 double，就是把默认的float32转成float64
            model_for_h = model_for_h.double()
            # 内层参数固定，不需要反向传播更新，所以去除梯度跟踪
            for param in model_for_h.parameters():
                param.requires_grad = False

            model_for_v = RFM_rep_a(d=2, J_n=J_n, x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max)
            model_for_v = model_for_v.apply(weights_init)
            # 将模型中所有的浮点数参数（权重、偏置等）转换为 double，就是把默认的float32转成float64
            model_for_v = model_for_v.double()
            # 内层参数固定，不需要反向传播更新，所以去除梯度跟踪
            for param in model_for_v.parameters():
                param.requires_grad = False

            model_x_t.append(model_for_h)
            model_x_t.append(model_for_v)
            model_x.append(model_x_t)

        models.append(model_x)
    return models


# Assembling the matrix A,f in linear system 'Au=f'
def assemble_matrix(models, points, M_p, J_n, Q):
    """
    models：模型
    points：配点列表
    M_p:划分的区间数（每个维度）
    Q：每个小区间取配点的个数(每个维度)，实际为Q+1
    J_n：线性层的输出维度
    """
    # 所有普通条件，因为方程个数是2，所以用配点数乘2
    A_I = np.zeros([2 * M_p * M_p * Q * Q, M_p * M_p * J_n])  # PDE term
    # 边界点
    A_B = np.zeros([4 * M_p * (Q + 1), M_p * M_p * J_n])  # boundary condition
    # 0阶连续性条件
    A_C_0 = np.zeros([M_p - 1, M_p * M_p * J_n])  # 0-order smoothness condition

    # 遍历每个区间，区间数M_p =========== start
    for i in range(M_p):
        for j in range(M_p):
            # 当前区间的配点
            point_for_h = torch.tensor(points[i][j], requires_grad=True)
            point_for_v = torch.tensor(points[i][j], requires_grad=True)
            # 当前单位分解区间的配点进入对应的神经网络
            h_out = models[i][j][0](point_for_h)
            h_values = h_out.detach().numpy()
            v_out = models[i][j][1](point_for_v)
            v_values = v_out.detach().numpy()

            # values = out.detach().numpy()
            # 当前区间的左边界和右边界的输出，注意神经网络的输出维度是J_n
            # value_l, value_r = values[0, :], values[-1, :]
            # 记录一阶导
            grad_h_x = []
            grad_h_t = []
            grad_v_x = []
            grad_v_t = []
            # 当前区间的配点求导，由于torch的局限，不能多维函数（J_n维）一起求导，所以只能循环
            for k in range(J_n):
                dh = torch.autograd.grad(outputs=h_out[:, k], inputs=point_for_h,
                                         grad_outputs=torch.ones_like(h_out[:, k]),
                                         create_graph=True, retain_graph=True)[0]

                dv = torch.autograd.grad(outputs=v_out[:, k], inputs=point_for_v,
                                         grad_outputs=torch.ones_like(v_out[:, k]),
                                         create_graph=True, retain_graph=True)[0]

                h_x = dh[:, 0]
                h_t = dh[:, 1]
                v_x = dv[:, 0]
                v_t = dv[:, 1]

                grad_h_x.append(h_x.detach().numpy())
                grad_h_t.append(h_t.detach().numpy())
                grad_v_x.append(v_x.detach().numpy())
                grad_v_t.append(v_t.detach().numpy())

            grad_h_x = np.array(grad_h_x).T
            grad_h_t = np.array(grad_h_t).T
            grad_v_x = np.array(grad_v_x).T
            grad_v_t = np.array(grad_v_t).T

            # 类似微分算子，注意这里不等同于微分算子，因为这里的Lu对于一个单点x，其输出维度是J_n
            Lu1 = grad_h_t + h_values * grad_v_x + v_values * grad_h_x
            Lu2 = h_values * grad_v_t + v_values * grad_h_t + 2 * h_values * v_values * grad_v_x + (
                    v_values ** 2 + h_values) * grad_h_x

            # 去掉右边界和上边界后剩的点的索引
            idx_filter = [m * (Q + 1) + n for m in range(Q) for n in range(Q)]

            # Lu = f condition
            # 构造分块对角矩阵，每个块Q行，J_n列，每个块就是一个小区间（扣除区间右端点）对应的方阵
            # i * M_p + j 表示当前是第几个区间
            # Q * Q * 2 该区间中用于计算的条件数，Q * Q 是配点数，因为是两个方程，所以条件数乘2
            A_I[(i * M_p + j) * Q * Q * 2:(i * M_p + j + 1) * Q * Q * 2, (i * M_p + j) * J_n:(i * M_p + j + 1) * J_n] \
                = np.vstack((Lu1[idx_filter], Lu2[idx_filter]))

            # boundary conditions
            # 整个定义域区间的左边界和右边界的输出，注意神经网络的输出维度是J_n
            # 构造的也是分块对角矩阵
            # 注意边界的输出不需要经过微分算子运算，因为是比对标签值
            # if i == 0:
            #     A_B[0, :J_n] = value_l
            # if i == M_p - 1:
            #     A_B[1, -J_n:] = value_r

            # smoothness conditions
            # 光滑性处理，当划分为多个区间时，区间连接的地方做光滑处理：
            # 每一行对应一个连接点，所以共M_p-1个，
            # 每行的数据由两部分组成：当前连接点作为左边区间的右端点负值和作为右边区间的左端点值
            # 最好的约束条件是：左边区间的右端点负值=作为右边区间的左端点值，但是这个条件不太好构建，
            # 所以构造一个弱约束：左边区间的右端点负值（经过外层聚合） + 右边区间的左端点值（经过外层聚合）= 0
            # 即想构造A=B，用A+B=0代替
            # if M_p > 1:
            #     if i == 0:
            #         print("n == 0")
            #         A_C_0[0, :J_n] = -value_r
            #     elif i == M_p - 1:
            #         print("n == M_p - 1")
            #         A_C_0[M_p - 2, -J_n:] = value_l
            #     else:
            #         print("0 < n < M_p - 1")
            #         A_C_0[i - 1, i * J_n:(i + 1) * J_n] = value_l
            #         A_C_0[i, i * J_n:(i + 1) * J_n] = -value_r
    # 遍历每个区间 =========== end

    if M_p > 1:
        A = np.concatenate((A_I, A_B, A_C_0, A_C_1), axis=0)
    else:
        A = np.concatenate((A_I, A_B), axis=0)

    # boundary conditions
    f[M_p * Q, :] = u(0.)
    f[M_p * Q + 1, :] = u(8.)

    return A, f


def main(M_p, J_n, Q):
    # prepare collocation points
    time_begin = time.time()
    points = []
    # 把定义域划分成多个子区间
    # 空间维度（x）
    for i in range(M_p):
        point_x = []
        # 时间维度(t)
        for j in range(M_p):
            # 单位分解子区间的左端(空间维度)
            x_min = (X_max - X_min) / M_p * i + X_min
            # 单位分解子区间的右端(空间维度)
            x_max = (X_max - X_min) / M_p * (i + 1) + X_min
            # 单位分解子区间的左端(时间维度)
            t_min = (T_max - T_min) / M_p * j + T_min
            # 单位分解子区间的右端(时间维度)
            t_max = (T_max - T_min) / M_p * (j + 1) + T_min
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
    print('Matrix shape: N=%s,M=%s' % A.shape)
    # rescaling
    # 这个缩放因子，对于其他方程，该如何确定？？？
    c = 100.0
    # 对每行按其最大值缩放
    # 为什么不按照绝对值的最大值缩放？这样会映射到[-c,c]，岂不完美？看文档应该是绝对值，代码错误？
    for i in range(len(A)):
        ratio = c / A[i, :].max()
        A[i, :] = A[i, :] * ratio
        f[i] = f[i] * ratio

    # solve
    # 求 Aw=f的最小二乘解，这里w就是外围参数，可以近似认为是神经网络的隐层到输出层的权重参数。
    w = lstsq(A, f)[0]

    # test
    error = test(models, M_p, J_n, Q, w)

    time_end = time.time()
    return error, time_end - time_begin


# analytical solution parameters
AA = 1
aa = 2.0 * np.pi
bb = 3.0 * np.pi
lamb = 4


def F(points, lamb):
    """
    真解构造右端项f
    """
    return


# calculate the l^{infty}-norm and l^{2}-norm error for u
def test(models, M_p, J_n, Q, w, plot=False):
    epsilon = []
    true_values = []
    numerical_values = []
    # 测试的时候，把网格变细，网格大小为配点的一半3
    test_Q = 2 * Q
    for n in range(M_p):
        # 这里的命名最好改为和之前的一致point，表示一个单位分解区间里面的配点
        points = torch.tensor(
            np.linspace((X_max - X_min) / M_p * n + X_min, (X_max - X_min) / M_p * (n + 1) + X_min, test_Q + 1),
            requires_grad=False).reshape([-1, 1])
        out = models[n](points)
        values = out.detach().numpy()
        true_value = vanal_u(points.numpy()).reshape([-1, 1])
        numerical_value = np.dot(values, w[n * J_n:(n + 1) * J_n, :])
        true_values.extend(true_value)
        numerical_values.extend(numerical_value)
        epsilon.extend(true_value - numerical_value)
    true_values = np.array(true_values)
    numerical_values = np.array(numerical_values)
    epsilon = np.array(epsilon)
    epsilon = np.maximum(epsilon, -epsilon)
    print('R_m=%s,M_p=%s,J_n=%s,Q=%s' % (R_m, M_p, J_n, Q))
    print('L_infty error =', epsilon.max(), ', L_2 error =', math.sqrt(8 * sum(epsilon * epsilon) / len(epsilon)))
    # 这行代码没用上，而且有问题，越界了
    x = [((X_max - X_min) / M_p) * i / test_Q for i in range(M_p * (test_Q + 1))]
    return math.sqrt((X_max - X_min) * sum(epsilon * epsilon) / len(epsilon))


if __name__ == '__main__':
    # 固定网络初始化参数，用于debug
    torch.manual_seed(123)
    # 超参数：随机特征（weight+bias）的均匀分布范围
    R_m = 3
    # 超参数：每个区间随机特征函数的个数，即每个区间对应的神经网络的隐层的维度
    J_n = 5  # the number of basis functions per PoU region
    # 超参数：每个区域配点的个数，其实配点个数是Q+1,这里的Q是每个单位分解区间的等分的区间数，注意这里针对的是每个维度
    Q = 3  # the number of collocation points per PoU region
    # 超参数：单位分解的区间数，注意这里指的是每个维度都划分为M_p个区间
    M_p = 2
    main(M_p, J_n, Q)
