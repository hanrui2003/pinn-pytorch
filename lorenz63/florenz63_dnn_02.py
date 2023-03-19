import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
from pyDOE import lhs
import math
import numpy as np


class LInfiniteLoss(nn.Module):
    def __init__(self):
        super(LInfiniteLoss, self).__init__()

    def forward(self, y, y_hat):
        max_loss = torch.max(torch.abs(y - y_hat))
        return max_loss


class FCN(nn.Module):
    """
    最大范数损失 + 输入正规化
    """

    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        # self.loss_func = nn.MSELoss(reduction='mean')
        self.loss_func = LInfiniteLoss()
        self.linears = nn.ModuleList([nn.Linear(layers[j], layers[j + 1]) for j in range(len(layers) - 1)])
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        # 正规化
        a = (x - x_lb) / (x_ub - x_lb)
        for j in range(len(self.linears) - 1):
            z = self.linears[j](a)
            a = self.activation(z)
        return self.linears[-1](a)

    def loss_bc(self, x_bc, y_bc):
        y_hat = self.forward(x_bc)
        return self.loss_func(y_bc, y_hat)

    def loss_pde(self, x_pde):
        y_hat = self.forward(x_pde)
        # 计算方程右端项 right-hand-side
        sum_item = discrete_c_all_train * y_hat
        # 分组求和，是一个聚合操作
        # 第一步，group：每个训练点对应的离散时间训练结果分为一组
        sum_item_group = sum_item.split(tuple(discrete_point_num_per_train))
        # 第二部，分组求和，使用map操作
        sum_per_train = list(map(lambda x: x.sum(0), sum_item_group))
        sum_per_train = torch.vstack(sum_per_train)

        # 右端项，lhs left-hand-side 因为和超立方采样lhs重名，所以家下划线
        lhs_ = gama_factor * sum_per_train

        x_nn = y_hat[key_train_index, [0]]
        y_nn = y_hat[key_train_index, [1]]
        z_nn = y_hat[key_train_index, [2]]

        dx_rhs = (sigma * (y_nn - x_nn)).unsqueeze(1)
        dy_rhs = (x_nn * (rho - z_nn) - y_nn).unsqueeze(1)
        dz_rhs = (x_nn * y_nn - beta * z_nn).unsqueeze(1)
        rhs_ = torch.hstack([dx_rhs, dy_rhs, dz_rhs])

        loss_pde = self.loss_func(lhs_, rhs_)

        return loss_pde

    def loss(self, x_bc, y_bc, x_pde):
        loss_bc = self.loss_bc(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        return loss_bc + loss_pde


if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    layers = [1, 32, 32, 3]
    PINN = FCN(layers)
    PINN.to(device)
    print(PINN)

    # 分数阶相关参数
    gama = torch.tensor(0.99)
    # 控制时间步长，大致理解为单位1长度，分割为200份
    lamda = torch.tensor(200.)

    # 总点数
    total_points = 301
    x_lb = torch.tensor(0.)
    x_ub = torch.tensor(1.5)

    # lorenz 方程参数
    rho = torch.tensor(28.0)
    sigma = torch.tensor(10.0)
    beta = torch.tensor(8.0 / 3.0)

    # 测试点，也是绘图点，并不是实际训练点，
    x_test = torch.linspace(x_lb, x_ub, total_points)

    obs = np.load('florenz63_rho_28_interval_0.005_total_point_301.npy')
    # 选取观测值
    idx = [1, 50, 100, 150, 200, 250, 300]
    x_train_bc = x_test[idx].unsqueeze(1)
    y_train_bc = torch.from_numpy(obs[idx])
    print('x_train_bc :', x_train_bc)
    print('y_train_bc :', y_train_bc)

    # 初值
    # x_train_bc = torch.tensor([[0.]])
    # y_train_bc = torch.tensor([[-8., 7., 27]])

    # 配置点
    n_f = 100
    x_train_nf = (x_lb + (x_ub - x_lb) * lhs(1, n_f)).float()
    x_train_nf = torch.vstack((x_train_bc, x_train_nf))
    print('x_train_nf :', x_train_nf)
    # 计算每个训练点做划分后，各有多少个小区间，对应公式中的lamda*t 的 向上取整
    interval_num_per_train = torch.ceil(lamda * x_train_nf).int().squeeze(1)
    # 计算每个训练点t,[0, t]离散化后，小区间的长度
    interval_len_per_train = x_train_nf.squeeze(1) / interval_num_per_train
    # 计算每个训练点做划分后，对应多少个离散点
    discrete_point_num_per_train = interval_num_per_train + 1
    # 真正关注的训练点，在整个训练向量中的索引值
    key_train_index = [sum(discrete_point_num_per_train[0:i + 1]) - 1 for i in range(len(discrete_point_num_per_train))]
    # 计算每个训练点t,对应的含Gamma项的因子
    gama_factor = 1. / (math.gamma(2 - gama) * interval_len_per_train ** gama).unsqueeze(1)
    # 计算每个训练点离散化后，对应的离散点，拼成一个大向量
    discrete_point_all_train = []
    # 计算每个训练点离散化后，对应的系数c，拼成一个大向量
    discrete_c_all_train = []

    for idx, n in enumerate(interval_num_per_train):
        for k in range(n + 1):
            if k == 0:
                discrete_point_all_train.append([0.])
                discrete_c_all_train.append([((n - 1) ** (1 - gama) - n ** (1 - gama)).item()])
            elif k == n:
                discrete_point_all_train.append([x_train_nf[idx].item()])
                discrete_c_all_train.append([1.])
            else:
                discrete_point_all_train.append([k * interval_len_per_train[idx].item()])
                discrete_c_all_train.append(
                    [((n - k + 1) ** (1 - gama) - 2 * (n - k) ** (1 - gama) + (n - k - 1) ** (1 - gama)).item()])

    x_train_bc = x_train_bc.float().to(device)
    y_train_bc = y_train_bc.float().to(device)
    # x_train_nf = x_train_nf.float().to(device)
    discrete_point_all_train = torch.tensor(discrete_point_all_train).to(device)
    discrete_c_all_train = torch.tensor(discrete_c_all_train).to(device)
    gama_factor = gama_factor.float().to(device)

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=40000,
                                                           verbose=True)

    # 训练网络
    epoch = 0
    while True:
        epoch += 1
        loss = PINN.loss(x_train_bc, y_train_bc, discrete_point_all_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 1000 == 0:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())

        if loss.item() < 0.01:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    PINN.cpu()
    nn_predict = PINN(x_test[:, None]).detach().numpy()
    x_nn = nn_predict[:, [0]]
    y_nn = nn_predict[:, [1]]
    z_nn = nn_predict[:, [2]]

    fig, ax = plt.subplots()
    ax.plot(x_test, x_nn, color='r', label='x')
    ax.plot(x_test, y_nn, color='g', label='y')
    ax.plot(x_test, z_nn, color='b', label='z')
    ax.set_title('Lorenz63')
    ax.set_xlabel('t', color='black')
    ax.set_ylabel('f(t)', color='black', rotation=0)
    ax.legend(loc='upper right')
    plt.savefig('./figure/florenz63_dnn_02_2d.png')
    plt.close()
    # plt.show()

    ax = plt.axes(projection="3d")
    ax.plot(x_nn, y_nn, z_nn, color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('lorenz63')
    plt.savefig('./figure/florenz63_dnn_02_3d.png')
    # plt.show()
