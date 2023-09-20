import torch
import torch.nn as nn
import numpy as np
import time
from fdm import error_plot, time_plot
import math
from scipy.linalg import lstsq
import matplotlib.pyplot as plt
from datetime import datetime
import torch.autograd as autograd

# computational domain，定义域的左右边界
X_min = 0.0
X_max = 8.0

AA = 1
aa = 2.0 * np.pi
bb = 3.0 * np.pi
lamb = 4


def u_tensor(x):
    """
    设置一个真解
    """
    return AA * torch.sin(bb * (x + 0.05)) * torch.cos(aa * (x + 0.05)) + 2.0


def d2u_dx2_tensor(x):
    """
    对u(x)的二阶导，就是拉普拉斯算子
    """
    return -AA * (aa * aa + bb * bb) * torch.sin(bb * (x + 0.05)) * torch.cos(aa * (x + 0.05)) \
           - 2.0 * AA * aa * bb * torch.cos(bb * (x + 0.05)) * torch.sin(aa * (x + 0.05))


def f_tensor(x):
    """
    逻辑与有限差分的一致，就是把变量换成tensor
    """
    return d2u_dx2_tensor(x) - lamb * u_tensor(x)


# 类似于自定义激活函数，定义基函数phi
# random feature basis when using \psi^{a} as PoU function
class RFM_rep_a(nn.Module):
    def __init__(self, d, J_n, x_max, x_min):
        """
        创建一个单隐层的全连接网络
        d:输入维度
        J_n:隐层维度，注意这里也是输出维度，因为后续没有聚合层。因为内层的参数固定，聚合层交给后续的外围参数优化。
        x_max: 单位分解区间的右端
        x_min: 单位分解区间的左端
        """
        super(RFM_rep_a, self).__init__()
        self.d = d
        self.J_n = J_n
        # 小区间半径，区间长度的一半
        self.r = (x_max - x_min) / 2.0
        # 小区间中心
        self.x_c = (x_max + x_min) / 2
        # 这里就是个简单的线性层+Tanh，注意这里是phi，不是psi
        first_linear = nn.Linear(self.d, self.J_n, bias=True)
        nn.init.uniform_(first_linear.weight, a=-R_m, b=R_m)
        nn.init.uniform_(first_linear.bias, a=-R_m, b=R_m)
        first_linear.weight.requires_grad = False
        first_linear.bias.requires_grad = False

        last_linear = nn.Linear(self.J_n, 1, bias=False)
        nn.init.uniform_(last_linear.weight, a=-R_m, b=R_m)
        # nn.init.uniform_(last_linear.bias, a=-R_m, b=R_m)

        self.phi = nn.Sequential(first_linear, nn.Tanh(), last_linear)

    def forward(self, x):
        # 标准化，使得取值在[-1,1]
        x = (x - self.x_c) / self.r
        x = self.phi(x)
        return x


class RFM_Net(nn.Module):
    def __init__(self, M_p, Q, J_n, X_max, X_min):
        super(RFM_Net, self).__init__()
        self.J_n = J_n
        self.M_p = M_p
        self.Q = Q
        self.loss_func = nn.MSELoss(reduction='mean')
        self.models = nn.ModuleList()
        for n in range(M_p):
            # 单位分解子区间的左端
            x_min = (X_max - X_min) / M_p * n + X_min
            # 单位分解子区间的右端
            x_max = (X_max - X_min) / M_p * (n + 1) + X_min
            model = RFM_rep_a(d=1, J_n=J_n, x_min=x_min, x_max=x_max)
            # 将模型中所有的浮点数参数（权重、偏置等）转换为 double，就是把默认的float32转成float64
            model = model.double()
            self.models.append(model)

    def forward(self, points):
        output = torch.empty((0, 1))
        for i in range(len(self.models)):
            output = torch.vstack((output, self.models[i](points[i * (Q + 1):(i + 1) * (Q + 1)])))
        return output

    def loss(self, x_train):
        x_train.requires_grad = True
        u = self.forward(x_train)
        u_x = autograd.grad(u, x_train, torch.ones_like(u), create_graph=True)[0]
        u_xx = autograd.grad(u_x, x_train, torch.ones_like(u_x), create_graph=True)[0]
        loss_pde = self.loss_func(u_xx - lamb * u, f_tensor(x_train))
        loss_bc = self.loss_func(u_tensor(x_train[[0, -1]]), u[[0, -1]])
        return loss_pde + loss_bc


def main(M_p, J_n, Q, lamb):
    # prepare collocation points
    time_begin = time.time()
    points = np.empty((0, 1))
    # 把定义域划分成多个子区间
    for n in range(M_p):
        # 单位分解子区间的左端
        x_min = (X_max - X_min) / M_p * n + X_min
        # 单位分解子区间的右端
        x_max = (X_max - X_min) / M_p * (n + 1) + X_min
        # 单位分解子区间的配点，Q表示每个单位分解子区间划分的子区间，所以算上端点，配点数为Q+1
        points = np.vstack((points, np.linspace(x_min, x_max, Q + 1).reshape([-1, 1])))

    x_train = torch.tensor(points)
    # prepare models
    # 一个单位分解子区间对应一个单隐层的全连接网络
    model = RFM_Net(M_p, Q, J_n, X_max, X_min)

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-2, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=1000,
                                                           verbose=True)

    # 记录训练开始时间
    start_time = datetime.now()
    print("Training started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    epoch = 0
    while True:
        epoch += 1
        loss = model.loss(x_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % 100 == 0:
            print(datetime.now(), 'epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
        if loss.item() < 1E-4:
            print(datetime.now(), 'epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'xxx-4.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
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
    # torch.manual_seed(123)
    # 公式中的lambda的平方
    lamb = 4
    # 超参数：随机特征（weight+bias）的均匀分布范围
    R_m = 3
    # 超参数：每个区间随机特征函数的个数，即每个区间对应的神经网络的隐层的维度
    J_n = 50  # the number of basis functions per PoU region
    # 超参数：每个区域配点的个数，其实配点个数是Q+1,这里的Q是每个单位分解区间的等分的区间数
    Q = 50  # the number of collocation points per PoU region
    main(4, J_n, Q, lamb)
    # RFM_Error = np.zeros([5, 3])
    # for i in range(5):  # the number of PoU regions
    #     # 划分的区间数
    #     M_p = 2 * (2 ** i)
    #     RFM_Error[i, 0] = int(M_p * J_n)
    #     RFM_Error[i, 1], RFM_Error[i, 2] = main(M_p, J_n, Q, lamb)
    # error_plot([RFM_Error])
    # time_plot([RFM_Error])
