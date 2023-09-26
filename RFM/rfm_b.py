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


# random feature basis when using \psi^{b} as PoU function
class RFM_rep_b(nn.Module):
    def __init__(self, d, J_n, x_max, x_min):
        super(RFM_rep_b, self).__init__()
        self.d = d
        self.J_n = J_n
        self.n = x_min / (X_max - X_min) * M_p
        self.r = (x_max - x_min) / 2.0
        self.x_c = (x_max + x_min) / 2
        self.phi = nn.Sequential(nn.Linear(self.d, self.J_n, bias=True), nn.Tanh())

    def forward(self, x):
        x = (x - self.x_c) / self.r
        y = self.phi(x)
        if self.n == 0:
            psi = ((x >= -1) & (x < 3 / 4)) * 1.0 + \
                  ((x >= 3 / 4) & (x < 5 / 4)) * (1 - torch.sin(2 * np.pi * x)) / 2 + \
                  ((x >= 5 / 4) | (x < -1)) * 0.
        elif self.n == M_p - 1:
            psi = ((x >= -5 / 4) & (x < -3 / 4)) * (1 + torch.sin(2 * np.pi * x)) / 2 + \
                  ((x >= -3 / 4) & (x <= 1)) * 1.0 + \
                  ((x > 1) | (x < -5 / 4)) * 0.
        else:
            psi = ((x >= -5 / 4) & (x < -3 / 4)) * (1 + torch.sin(2 * np.pi * x)) / 2 + \
                  ((x >= -3 / 4) & (x < 3 / 4)) * 1.0 + \
                  ((x >= 3 / 4) & (x < 5 / 4)) * (1 - torch.sin(2 * np.pi * x)) / 2 + \
                  ((x >= 5 / 4) | (x < -5 / 4)) * 0.

        return psi * y


# predefine the random feature functions in each PoU region
def pre_define(M_p, J_n):
    """
    为每个单位分解子区间生成一个神经网络
    M_p：单位分解区间的数量，对应着单隐层神经网络的数量
    J_n：神经网络隐层的维度
    """
    models = []
    for n in range(M_p):
        # 单位分解子区间的左端
        x_min = (X_max - X_min) / M_p * n + X_min
        # 单位分解子区间的右端
        x_max = (X_max - X_min) / M_p * (n + 1) + X_min
        model = RFM_rep_b(d=1, J_n=J_n, x_min=x_min, x_max=x_max)
        model = model.apply(weights_init)
        # 将模型中所有的浮点数参数（权重、偏置等）转换为 double，就是把默认的float32转成float64
        model = model.double()
        # 内层参数固定，不需要反向传播更新，所以去除梯度跟踪
        for param in model.parameters():
            param.requires_grad = False
        models.append(model)
    return models


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
    A_I = np.zeros([M_p * Q + 1, M_p * J_n])  # PDE term
    # 左右两个边界点，所以横轴维度为2
    # 每个边界点经过一个神经网络的输出是J_n维，一共M_p个网络，所以纵轴维度是M_p * J_n
    A_B = np.zeros([2, M_p * J_n])  # boundary condition

    f = np.zeros([M_p * Q + 3, 1])

    # 遍历每个区间，区间数M_p =========== start
    Lu = np.empty((len(points), 0))
    for n in range(M_p):
        # forward and grad
        # 当前区间的配点
        point = torch.tensor(points, requires_grad=True)
        # 当前单位分解区间的配点进入对应的神经网络
        out = models[n](point)
        values = out.detach().numpy()
        # 当前区间的左边界和右边界的输出，注意神经网络的输出维度是J_n
        value_l, value_r = values[0, :], values[-1, :]
        # 记录一阶导和二阶导
        grad2 = []
        # 当前区间的配点求导，由于torch的局限，不能多维函数（J_n维）一起求导，所以只能循环
        for i in range(J_n):
            g1 = torch.autograd.grad(outputs=out[:, i], inputs=point,
                                     grad_outputs=torch.ones_like(out[:, i]),
                                     create_graph=True, retain_graph=True)[0]

            g2 = torch.autograd.grad(outputs=g1[:, 0], inputs=point,
                                     grad_outputs=torch.ones_like(out[:, i]),
                                     create_graph=False, retain_graph=True)[0]

            grad2.append(g2.squeeze().detach().numpy())
        grad2 = np.array(grad2).T

        # 类似微分算子，注意这里不等同于微分算子，因为这里的Lu对于一个单点x，其输出维度是J_n
        # 这里代码可能有点问题，跟pinn.py一样的错误，把加写成了减，
        # 但是和下面f的计算（也是减）是对应的，将错就错还是说文档错误？
        # 无伤大雅，因为都改为加，运行也是一样的收敛效果。
        Lu = grad2 - lamb * values

        # Lu = f condition
        A_I[:, n * J_n:(n + 1) * J_n] = Lu
        # # 真解构造f，因为方程中认为f是已知的，这里也是一个小区间（扣除区间右端点）对应的向量
        # f[n * Q:(n + 1) * Q, :] = F(points[n], lamb).reshape([-1, 1])[:Q]
        #
        # # boundary conditions
        # # 整个定义域区间的左边界和右边界的输出，注意神经网络的输出维度是J_n
        # # 构造的也是分块对角矩阵
        # # 注意边界的输出不需要经过微分算子运算，因为是比对标签值
        # if n == 0:
        #     A_B[0, :J_n] = value_l
        # if n == M_p - 1:
        #     A_B[1, -J_n:] = value_r

    # 遍历每个区间 =========== end

    A = np.concatenate((A_I, A_B), axis=0)

    # boundary conditions
    f[M_p * Q, :] = u(0.)
    f[M_p * Q + 1, :] = u(8.)

    return A, f


def main(M_p, J_n, Q, lamb):
    # prepare collocation points
    time_begin = time.time()
    points = np.linspace(X_min, X_max, M_p * Q + 1).reshape([-1, 1])

    # prepare models
    # 一个单位分解子区间对应一个单隐层的全连接网络
    models = pre_define(M_p, J_n)

    # matrix define (Au=f)
    A, f = assemble_matrix(models, points, M_p, J_n, Q, lamb)
    print('***********************')
    print('Matrix shape: N=%s,M=%s' % A.shape)
    # rescaling
    # 这个缩放因子，对于其他方程，该如何确定？？？
    c = 100.0
    # 对每行按其最大值缩放
    # 为什么不按照绝对值的最大值缩放？这样会映射到[-c,c]，岂不完美？
    # 看文档应该是绝对值，这里应该是代码错误，因为当最大值是0的时候，异常。确实也遇到了这种问题。
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

vanal_u = np.vectorize(u)
vanal_d2u_dx2 = np.vectorize(d2u_dx2)


def F(points, lamb):
    """
    真解构造右端项f
    """
    return vanal_d2u_dx2(points) - lamb * vanal_u(points)


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
    # 公式中的lambda的平方
    lamb = 4
    # 超参数：随机特征（weight+bias）的均匀分布范围
    R_m = 3
    # 超参数：每个区间随机特征函数的个数，即每个区间对应的神经网络的隐层的维度
    J_n = 5  # the number of basis functions per PoU region
    # 超参数：每个区域配点的个数，其实配点个数是Q+1,这里的Q是每个单位分解区间的等分的区间数
    Q = 5  # the number of collocation points per PoU region
    M_p = 2
    main(M_p, J_n, Q, lamb)
    # RFM_Error = np.zeros([5, 3])
    # for i in range(5):  # the number of PoU regions
    #     # 划分的区间数
    #     M_p = 2 * (2 ** i)
    #     RFM_Error[i, 0] = int(M_p * J_n)
    #     RFM_Error[i, 1], RFM_Error[i, 2] = main(M_p, J_n, Q, lamb)
    # error_plot([RFM_Error])
    # time_plot([RFM_Error])
