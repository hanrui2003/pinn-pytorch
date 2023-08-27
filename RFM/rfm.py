import torch
import torch.nn as nn
import numpy as np
import time
from fdm import u, d2u_dx2, error_plot, time_plot
import math
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

# computational domain，定义域的左右边界
X_min = 0.0
X_max = 8.0


# random initialization for parameters
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a=-R_m, b=R_m)
        nn.init.uniform_(m.bias, a=-R_m, b=R_m)


# 类似于自定义激活函数，定义基函数phi
# random feature basis when using \psi^{a} as PoU function
class RFM_rep_a(nn.Module):
    def __init__(self, d, J_n, x_max, x_min):
        super(RFM_rep_a, self).__init__()
        # 输入维度
        self.d = d
        # 输出维度
        self.J_n = J_n
        # 小区间半径，区间长度的一半
        self.r = (x_max - x_min) / 2.0
        # 小区间中心
        self.x_c = (x_max + x_min) / 2
        # 这里就是个简单的线性层+Tanh
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
        return (psi * y)


# predefine the random feature functions in each PoU region
def pre_define(M_p, J_n):
    models = []
    for n in range(M_p):
        x_min = (X_max - X_min) / M_p * n + X_min
        x_max = (X_max - X_min) / M_p * (n + 1) + X_min
        model = RFM_rep_a(d=1, J_n=J_n, x_min=x_min, x_max=x_max)
        model = model.apply(weights_init)
        # 将模型中所有的浮点数参数（权重、偏置等）转换为 double
        model = model.double()
        for param in model.parameters():
            param.requires_grad = False
        models.append(model)
    return (models)


# Assembling the matrix A,f in linear system 'Au=f'
def assemble_matrix(models, points, M_p, J_n, Q, lamb):
    """
    models：模型，其数量等于M_p
    points：配点列表，配点的总数是M_p*(Q+1)
    M_p:划分的区间数
    Q：每个小区间取配点的个数，实际为Q+1
    J_n：线性层的输出维度
    lamb：公式中lambda的平方
    """
    A_I = np.zeros([M_p * Q, M_p * J_n])  # PDE term
    A_B = np.zeros([2, M_p * J_n])  # boundary condition
    A_C_0 = np.zeros([M_p - 1, M_p * J_n])  # 0-order smoothness condition
    A_C_1 = np.zeros([M_p - 1, M_p * J_n])  # 1-order smoothness condition
    f = np.zeros([M_p * Q + 2 * (M_p - 1) + 2, 1])

    # 遍历每个区间，区间数M_p
    for n in range(M_p):
        # forward and grad
        # 当前区间的配点
        point = torch.tensor(points[n], requires_grad=True)
        out = models[n](point)
        values = out.detach().numpy()
        # 小区间的左边界和右边界
        value_l, value_r = values[0, :], values[-1, :]
        # 记录一阶导和二阶导
        grad1 = []
        grad2 = []
        # 每个区间的配点求导，由于torch的局限，不能多维函数一起求导，所以只能循环
        for i in range(J_n):
            g1 = torch.autograd.grad(outputs=out[:, i], inputs=point,
                                     grad_outputs=torch.ones_like(out[:, i]),
                                     create_graph=True, retain_graph=True)[0]
            grad1.append(g1.squeeze().detach().numpy())

            g2 = torch.autograd.grad(outputs=g1[:, 0], inputs=point,
                                     grad_outputs=torch.ones_like(out[:, i]),
                                     create_graph=False, retain_graph=True)[0]
            grad2.append(g2.squeeze().detach().numpy())
        grad1 = np.array(grad1).T
        grad2 = np.array(grad2).T
        grad_l = grad1[0, :]
        grad_r = grad1[-1, :]
        # 微分算子，不过这里和公式不太对应，应该是加
        # 但是和下面f的计算（也是减）是对应的，将错就错还是说文档错误？
        Lu = grad2 - lamb * values

        # Lu = f condition
        # 构造分块对角矩阵，每个块就是一个小区间（扣除尾部）对应的方阵
        A_I[n * Q:(n + 1) * Q, n * J_n:(n + 1) * J_n] = Lu[:Q, :]
        f[n * Q:(n + 1) * Q, :] = F(points[n], lamb).reshape([-1, 1])[:Q]

        # boundary conditions
        # 用分块对角阵表示边值
        if n == 0:
            A_B[0, :J_n] = value_l
        if n == M_p - 1:
            A_B[1, -J_n:] = value_r

        # smoothness conditions
        # 这里需要再细看，请教老师
        if M_p > 1:
            if n == 0:
                A_C_0[0, :J_n] = -value_r
                A_C_1[0, :J_n] = -grad_r
            elif n == M_p - 1:
                A_C_0[M_p - 2, -J_n:] = value_l
                A_C_1[M_p - 2, -J_n:] = grad_l
            else:
                A_C_0[n - 1, n * J_n:(n + 1) * J_n] = value_l
                A_C_1[n - 1, n * J_n:(n + 1) * J_n] = grad_l
                A_C_0[n, n * J_n:(n + 1) * J_n] = -value_r
                A_C_1[n, n * J_n:(n + 1) * J_n] = -grad_r
    if M_p > 1:
        A = np.concatenate((A_I, A_B, A_C_0, A_C_1), axis=0)
    else:
        A = np.concatenate((A_I, A_B), axis=0)

    # boundary conditions
    f[M_p * Q, :] = u(0.)
    f[M_p * Q + 1, :] = u(8.)

    # 为什么f不赋光滑条件，还是说光滑条件就是0

    return (A, f)


def main(M_p, J_n, Q, lamb):
    # prepare collocation points
    time_begin = time.time()
    points = []
    for n in range(M_p):
        x_min = (X_max - X_min) / M_p * n + X_min
        x_max = (X_max - X_min) / M_p * (n + 1) + X_min
        points.append(np.linspace(x_min, x_max, Q + 1).reshape([-1, 1]))

    # prepare models
    models = pre_define(M_p, J_n)

    # matrix define (Au=f)
    A, f = assemble_matrix(models, points, M_p, J_n, Q, lamb)
    print('***********************')
    print('Matrix shape: N=%s,M=%s' % (A.shape))
    # rescaling
    c = 100.0
    for i in range(len(A)):
        ratio = c / A[i, :].max()
        A[i, :] = A[i, :] * ratio
        f[i] = f[i] * ratio

    # solve
    w = lstsq(A, f)[0]

    # test
    error = test(models, M_p, J_n, Q, w)

    time_end = time.time()
    return (error, time_end - time_begin)


# analytical solution parameters
AA = 1
aa = 2.0 * np.pi
bb = 3.0 * np.pi
lamb = 4

vanal_u = np.vectorize(u)
vanal_d2u_dx2 = np.vectorize(d2u_dx2)


def F(points, lamb):
    """
    真解构造右端项f
    """
    return (vanal_d2u_dx2(points) - lamb * vanal_u(points))


# calculate the l^{infty}-norm and l^{2}-norm error for u
def test(models, M_p, J_n, Q, w, plot=False):
    epsilon = []
    true_values = []
    numerical_values = []
    test_Q = 2 * Q
    for n in range(M_p):
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
    x = [((X_max - X_min) / M_p) * i / test_Q for i in range(M_p * (test_Q + 1))]
    return (math.sqrt((X_max - X_min) * sum(epsilon * epsilon) / len(epsilon)))


if __name__ == '__main__':
    lamb = 4
    R_m = 3
    J_n = 50  # the number of basis functions per PoU region
    Q = 50  # the number of collocation pointss per PoU region
    RFM_Error = np.zeros([5, 3])
    for i in range(5):  # the number of PoU regions
        M_p = 2 * (2 ** i)
        RFM_Error[i, 0] = int(M_p * J_n)
        RFM_Error[i, 1], RFM_Error[i, 2] = main(M_p, J_n, Q, lamb)
    error_plot([RFM_Error])
    time_plot([RFM_Error])
