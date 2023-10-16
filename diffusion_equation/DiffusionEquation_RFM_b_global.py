import torch
import torch.nn as nn
import numpy as np
import time
import math
from scipy.linalg import lstsq, block_diag
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)

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


class RFM_rep_b_global(nn.Module):
    def __init__(self, d, M):
        super(RFM_rep_b_global, self).__init__()
        self.d = d
        self.M = M
        self.phi = nn.Sequential(nn.Linear(self.d, self.M, bias=True), nn.Tanh())

    def forward(self, x):
        y = self.phi(x)
        return y


# random feature basis when using \psi^{b} as PoU function
class RFM_rep_b(nn.Module):
    def __init__(self, d, J_n, x_max, x_min, t_min, t_max, n_x, n_t, M_p):
        """
        :param n_x : 空间维度，第几个区间
        :param n_t : 时间维度，第几个区间
        """
        super(RFM_rep_b, self).__init__()
        self.d = d
        self.J_n = J_n
        self.n_x = n_x
        self.n_t = n_t
        self.M_p = M_p
        # 小区间半径，区间长度的一半
        self.r = torch.tensor(((x_max - x_min) / 2.0, (t_max - t_min) / 2.0))
        # 小区间中心
        self.y_c = torch.tensor(((x_max + x_min) / 2, (t_max + t_min) / 2))
        self.phi = nn.Sequential(nn.Linear(self.d, self.J_n, bias=True), nn.Tanh())

    def forward(self, y):
        # 标准化，使得取值在[-1,1]
        y = (y - self.y_c) / self.r
        z = self.phi(y)
        x = y[:, [0]]
        t = y[:, [1]]
        # 注意，默认情况psi会自动复制为0，所以分段函数不用写otherwise
        if self.n_x == 0:
            psi_x = ((x >= -1) & (x < 3 / 4)) * 1.0 + \
                    ((x >= 3 / 4) & (x < 5 / 4)) * (1 - torch.sin(2 * np.pi * x)) / 2
        elif self.n_x == self.M_p[0] - 1:
            psi_x = ((x >= -5 / 4) & (x < -3 / 4)) * (1 + torch.sin(2 * np.pi * x)) / 2 + \
                    ((x >= -3 / 4) & (x <= 1)) * 1.0
        else:
            psi_x = ((x >= -5 / 4) & (x < -3 / 4)) * (1 + torch.sin(2 * np.pi * x)) / 2 + \
                    ((x >= -3 / 4) & (x < 3 / 4)) * 1.0 + \
                    ((x >= 3 / 4) & (x < 5 / 4)) * (1 - torch.sin(2 * np.pi * x)) / 2

        if self.n_t == 0:
            psi_t = ((t >= -1) & (t < 3 / 4)) * 1.0 + \
                    ((t >= 3 / 4) & (t < 5 / 4)) * (1 - torch.sin(2 * np.pi * t)) / 2
        elif self.n_t == self.M_p[1] - 1:
            psi_t = ((t >= -5 / 4) & (t < -3 / 4)) * (1 + torch.sin(2 * np.pi * t)) / 2 + \
                    ((t >= -3 / 4) & (t <= 1)) * 1.0
        else:
            psi_t = ((t >= -5 / 4) & (t < -3 / 4)) * (1 + torch.sin(2 * np.pi * t)) / 2 + \
                    ((t >= -3 / 4) & (t < 3 / 4)) * 1.0 + \
                    ((t >= 3 / 4) & (t < 5 / 4)) * (1 - torch.sin(2 * np.pi * t)) / 2

        return psi_x * psi_t * z


# predefine the random feature functions in each PoU region
def pre_define(M_p, M, J_n):
    """
    为每个单位分解子区间生成一个神经网络
    M_p：单位分解区间的数量(M_p[0]*M_p[1])，对应着单隐层神经网络的数量
    J_n：神经网络隐层的维度
    """

    # 全局
    global_model = RFM_rep_b_global(d=2, M=M)
    global_model = global_model.apply(weights_init)
    for param in global_model.parameters():
        param.requires_grad = False

    # 局部
    local_models = []
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
            model = RFM_rep_b(d=2, J_n=J_n, x_min=x_min, x_max=x_max, t_min=t_min, t_max=t_max, n_x=i, n_t=j, M_p=M_p)
            model = model.apply(weights_init)
            # 内层参数固定，不需要反向传播更新，所以去除梯度跟踪
            for param in model.parameters():
                param.requires_grad = False

            model_x.append(model)

        local_models.append(model_x)
    return global_model, local_models


# Assembling the matrix A,f in linear system 'Au=f'
def assemble_matrix(global_model, local_models, M_p, M, J_n, Q, pde_points, ic_points, bc_points):
    """
    models：模型
    points：配点列表
    M_p:划分的区间数,M_p[0]*M_p[1]
    Q：每个小区间取配点的个数(每个维度)，实际为Q+1
    J_n：线性层的输出维度
    """
    print(datetime.now(), "assemble_matrix start")

    # 全局计算
    pde_point_G = torch.tensor(pde_points, requires_grad=True)
    ic_point_G = torch.tensor(ic_points, requires_grad=True)
    bc_point_G = torch.tensor(bc_points, requires_grad=True)

    # 当前单位分解区间的配点进入对应的神经网络
    pde_out_G = global_model(pde_point_G)
    pde_values_G = pde_out_G.detach().numpy()

    ic_out_G = global_model(ic_point_G)
    ic_values_G = ic_out_G.detach().numpy()

    bc_out_G = global_model(bc_point_G)
    bc_values_G = bc_out_G.detach().numpy()

    # 记录一阶导
    grad_u_t_G = []
    grad_u_xx_G = []
    # 当前区间的配点求导，由于torch的局限，不能多维函数（J_n维）一起求导，所以只能循环
    for k in range(M):
        u_x_t_G = torch.autograd.grad(outputs=pde_out_G[:, k], inputs=pde_point_G,
                                      grad_outputs=torch.ones_like(pde_out_G[:, k]),
                                      create_graph=True, retain_graph=True)[0]

        u_xx_tt_G = torch.autograd.grad(outputs=u_x_t_G, inputs=pde_point_G,
                                        grad_outputs=torch.ones_like(u_x_t_G),
                                        create_graph=True, retain_graph=True)[0]

        u_t_G = u_x_t_G[:, 1]
        u_xx_G = u_xx_tt_G[:, 0]

        grad_u_t_G.append(u_t_G.detach().numpy())
        grad_u_xx_G.append(u_xx_G.detach().numpy())

    grad_u_t_G = np.array(grad_u_t_G).T
    grad_u_xx_G = np.array(grad_u_xx_G).T

    A_P_G = grad_u_xx_G - grad_u_t_G
    # 初值条件
    A_I_G = ic_values_G
    # 边界条件
    A_B_G = bc_values_G

    A_G = np.concatenate((A_P_G, A_I_G, A_B_G), axis=0)

    # 局部计算
    # 所有PDE条件
    A_P_L = np.zeros((len(pde_points), M_p[0] * M_p[1] * J_n))
    # 初值条件
    A_I_L = np.zeros((len(ic_points), M_p[0] * M_p[1] * J_n))
    # 边界条件
    A_B_L = np.zeros((len(bc_points), M_p[0] * M_p[1] * J_n))

    pde_point_L = torch.tensor(pde_points, requires_grad=True)
    ic_point_L = torch.tensor(ic_points, requires_grad=True)
    bc_point_L = torch.tensor(bc_points, requires_grad=True)

    # 遍历每个区间，区间数M_p =========== start
    for i in range(M_p[0]):
        for j in range(M_p[1]):
            # 当前单位分解区间的配点进入对应的神经网络
            pde_out_L = local_models[i][j](pde_point_L)
            pde_values_L = pde_out_L.detach().numpy()

            ic_out_L = local_models[i][j](ic_point_L)
            ic_values_L = ic_out_L.detach().numpy()

            bc_out_L = local_models[i][j](bc_point_L)
            bc_values_L = bc_out_L.detach().numpy()

            # 记录一阶导
            grad_u_t_L = []
            grad_u_xx_L = []
            # 当前区间的配点求导，由于torch的局限，不能多维函数（J_n维）一起求导，所以只能循环
            for k in range(J_n):
                u_x_t_L = torch.autograd.grad(outputs=pde_out_L[:, k], inputs=pde_point_L,
                                              grad_outputs=torch.ones_like(pde_out_L[:, k]),
                                              create_graph=True, retain_graph=True)[0]

                u_xx_tt_L = torch.autograd.grad(outputs=u_x_t_L, inputs=pde_point_L,
                                                grad_outputs=torch.ones_like(u_x_t_L),
                                                create_graph=True, retain_graph=True)[0]

                u_t_L = u_x_t_L[:, 1]
                u_xx_L = u_xx_tt_L[:, 0]

                grad_u_t_L.append(u_t_L.detach().numpy())
                grad_u_xx_L.append(u_xx_L.detach().numpy())

            grad_u_t_L = np.array(grad_u_t_L).T
            grad_u_xx_L = np.array(grad_u_xx_L).T

            # 类似微分算子，注意这里不等同于微分算子，因为这里的Lu对于一个单点x，其输出维度是J_n
            Lu = grad_u_xx_L - grad_u_t_L

            # Lu = f condition
            A_P_L[:, (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = Lu

            # 初值条件
            A_I_L[:, (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = ic_values_L

            # 边界条件
            A_B_L[:, (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = bc_values_L

    # 遍历每个区间 =========== end
    print("assemble A and f")
    A_L = np.concatenate((A_P_L, A_I_L, A_B_L), axis=0)
    A = np.concatenate((A_G, A_L), axis=1)

    f_P = np.exp(-pde_points[:, [1]]) * (1 - np.pi ** 2) * np.sin(np.pi * pde_points[:, [0]])
    f_I = np.sin(np.pi * ic_points[:, [0]])
    f_B = np.zeros((len(bc_points), 1))
    f = np.concatenate((f_P, f_I, f_B), axis=0)

    print(datetime.now(), "assemble_matrix end")
    return A, f


def main(M_p, M, J_n, Q):
    # prepare collocation points
    time_begin = time.time()
    x = np.linspace(X_min, X_max, M_p[0] * Q + 1)
    t = np.linspace(T_min, T_max, M_p[1] * Q + 1)
    X, T = np.meshgrid(x, t)
    pde_points = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    ic_points = np.hstack((X[0][:, None], T[0][:, None]))
    left_bc_points = np.hstack((X[:, 0][:, None], T[:, 0][:, None]))
    right_bc_points = np.hstack((X[:, -1][:, None], T[:, -1][:, None]))
    bc_points = np.vstack((left_bc_points, right_bc_points))

    # prepare models
    # 一个单位分解子区间对应一个单隐层的全连接网络
    global_model, local_models = pre_define(M_p, M, J_n)

    # matrix define (Au=f)
    A, f = assemble_matrix(global_model, local_models, M_p, M, J_n, Q, pde_points, ic_points, bc_points)
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

    # # 定义目标函数，即要最小化的函数
    # def objective_function(x, A, f):
    #     equation1 = np.dot(A, x) - f
    #     return np.linalg.norm(equation1)
    #
    # # 定义初始猜测值
    # w0 = np.random.uniform(low=-R_m, high=R_m, size=M_p[0] * M_p[1] * J_n)
    #
    # result = minimize(objective_function, w0, args=(A, f.squeeze(1)), method='BFGS')
    #
    # w = result.x[:, None]

    # test
    test(global_model, local_models, M_p, M, J_n, Q, w)


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


def plot_err(X1, T1, U1):
    """
    误差分布
    """
    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(12, 6))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    cp1 = ax1.contourf(T1, X1, U1, 20, cmap="rainbow")
    fig.colorbar(cp1, ax=ax1)
    ax1.set_title('err')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')

    ax2.plot_surface(T1, X1, U1, cmap="rainbow")
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('err')

    plt.show()


# calculate the l^{infinity}-norm and l^{2}-norm error for u
def test(global_model, local_models, M_p, M, J_n, Q, w):
    # 测试的时候，把网格变细，网格大小为配点的一半3
    test_Q = 2 * Q
    x = np.linspace(X_min, X_max, M_p[0] * test_Q + 1)
    t = np.linspace(T_min, T_max, M_p[1] * test_Q + 1)
    X, T = np.meshgrid(x, t)
    points = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    point = torch.tensor(points, requires_grad=False)

    A_G = global_model(point).detach().numpy()
    A_L = np.zeros((len(point), M_p[0] * M_p[1] * J_n))
    for i in range(M_p[0]):
        for j in range(M_p[1]):
            out = local_models[i][j](point)
            values = out.detach().numpy()
            A_L[:, (i * M_p[1] + j) * J_n:(i * M_p[1] + j + 1) * J_n] = values

    A = np.concatenate((A_G, A_L), axis=1)
    numerical_values = np.dot(A, w)
    true_values = f_real(points[:, [0]], points[:, [1]])
    epsilon = true_values - numerical_values
    # 就是取绝对值操作
    epsilon = np.maximum(epsilon, -epsilon)

    L_inf = epsilon.max()
    L_2 = math.sqrt(sum(epsilon ** 2) / len(epsilon))
    relative_error = np.linalg.norm(numerical_values - true_values) / np.linalg.norm(true_values)

    print('R_m=%s,M_p=%s,M=%s,J_n=%s,Q=%s' % (R_m, M_p, M, J_n, Q))
    print('L_infty error =', L_inf, ', L_2 error =', L_2, ', relative error =', relative_error)

    U_true = true_values.reshape((X.shape[0], X.shape[1]))
    U_numerical = numerical_values.reshape((X.shape[0], X.shape[1]))

    plot(X, T, U_true, X, T, U_numerical)
    plot_err(X, T, np.abs(U_true - U_numerical))

    return L_2


if __name__ == '__main__':
    # 固定网络初始化参数，用于debug
    # torch.manual_seed(123)
    # 超参数：随机特征（weight+bias）的均匀分布范围
    R_m = 2
    M = 200
    # 超参数：每个区间随机特征函数的个数，即每个区间对应的神经网络的隐层的维度
    J_n = 50  # the number of basis functions per PoU region
    # 超参数：每个区域配点的个数，其实配点个数是Q+1,这里的Q是每个单位分解区间的等分的区间数，注意这里针对的是每个维度
    Q = 50  # the number of collocation points per PoU region
    # 超参数：单位分解的区间数，注意这里指的是每个维度都划分为M_p个区间
    M_p = 4, 2
    main(M_p, M, J_n, Q)
