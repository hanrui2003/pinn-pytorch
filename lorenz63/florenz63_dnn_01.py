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
        self.loss_func = nn.MSELoss(reduction='mean')
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
        x_nn = y_hat[:, [0]]
        y_nn = y_hat[:, [1]]
        z_nn = y_hat[:, [2]]

        dx_rhs = sigma * (y_nn - x_nn)
        dy_rhs = x_nn * (rho - z_nn) - y_nn
        dz_rhs = x_nn * y_nn - beta * z_nn
        rhs_ = torch.hstack([dx_rhs, dy_rhs, dz_rhs])
        lhs_ = torch.empty_like(rhs_)

        for idx, t in enumerate(x_pde):
            n = torch.tensor(torch.ceil(lamda * t), dtype=torch.int32)
            delta_t = t / n
            fdm = y_hat[0].clone()
            fdm -= (n ** (1 - gama) - (n - 1) ** (1 - gama)) * self.forward(torch.zeros_like(t))
            for k in range(1, n):
                c = ((n - k + 1) ** (1 - gama) - (n - k) ** (1 - gama)) \
                    - ((n - k) ** (1 - gama) - (n - k - 1) ** (1 - gama))
                u = self.forward(k * delta_t)
                fdm += c * u
            gama_factor = 1 / (math.gamma(2 - gama) * (delta_t ** gama))
            fdm *= gama_factor
            lhs_[idx] = fdm

        loss_pde = self.loss_func(lhs_, rhs_)
        # n = torch.ceil(lamda * x_pde)
        # delta_t = x_pde / n
        # fdm = y_hat.clone()
        # fdm -= (n ** (1 - gama) - (n - 1) ** (1 - gama)) * self.forward(torch.zeros_like(x_pde))
        # for k in range(1, n):
        #     c = ((n - k + 1) ** (1 - gama) - (n - k) ** (1 - gama)) \
        #         - ((n - k) ** (1 - gama) - (n - k - 1) ** (1 - gama))
        #     u = self.forward(k * delta_t)
        #     fdm += c * u
        # gama_factor = 1 / (math.gamma(2 - gama) * (delta_t ** gama))
        # fdm *= gama_factor
        #
        # rhs = torch.tensor([sigma * (y_nn - x_nn), x_nn * (rho - z_nn) - y_nn, x_nn * y_nn - beta * z_nn])
        #
        # loss_pde = self.loss_func(fdm, rhs)

        return loss_pde

    def loss(self, x_bc, y_bc, x_pde):
        loss_bc = self.loss_bc(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        loss_max = loss_bc if loss_bc > loss_pde else loss_pde
        return loss_max


if "__main__" == __name__:
    torch.manual_seed(1234)
    np.random.seed(1234)

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    layers = [1, 64, 64, 3]
    PINN = FCN(layers)
    PINN.to(device)
    print(PINN)

    # 分数阶相关参数
    gama = torch.tensor(0.6)
    # 控制时间步长，大致理解为单位1长度，分割为200份
    lamda = torch.tensor(200.)

    # 总点数
    total_points = 200
    # 防止采样到边界点0，这样有限差分会有问题（分母为0）
    x_lb = torch.tensor(0.00000001)
    x_ub = torch.tensor(3.)

    # lorenz 方程参数
    rho = torch.tensor(28.0)
    sigma = torch.tensor(10.0)
    beta = torch.tensor(8.0 / 3.0)

    # 测试点，也是绘图点
    x_test = torch.linspace(x_lb, x_ub, total_points)

    # 初值
    x_train_bc = torch.tensor([[0.]])
    y_train_bc = torch.tensor([[1.508870, -1.531271, 25.46091]])

    # 配置点
    n_f = total_points
    x_train_nf = (x_lb + (x_ub - x_lb) * lhs(1, n_f)).float()

    x_train_bc = x_train_bc.float().to(device)
    y_train_bc = y_train_bc.float().to(device)
    x_train_nf = x_train_nf.float().to(device)

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=40000,
                                                           verbose=True)

    # 训练网络
    epoch = 0
    while True:
        epoch += 1
        loss = PINN.loss(x_train_bc, y_train_bc, x_train_nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 100 == 0:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())

        if loss.item() < 0.1:
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
    ax.set_title('Lorenz with 300 point')
    ax.set_xlabel('t', color='black')
    ax.set_ylabel('f(t)', color='black', rotation=0)
    ax.legend(loc='upper right')
    plt.savefig('./figure/florenz63_dnn_01_2d.png')
    plt.close()
    # plt.show()

    ax = plt.axes(projection="3d")
    ax.plot(x_nn, y_nn, z_nn, color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('lorenz63')
    plt.savefig('./figure/florenz63_dnn_01_3d.png')
    # plt.show()
