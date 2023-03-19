import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs


class LInfiniteLoss(nn.Module):
    def __init__(self):
        super(LInfiniteLoss, self).__init__()

    def forward(self, y, y_hat):
        max_loss = torch.max(torch.abs(y - y_hat))
        return max_loss


class FCN(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_func = LInfiniteLoss()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        # 正规化
        a = (x - x_lb) / (x_ub - x_lb)
        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)
        return self.linears[-1](a)

    def loss_bc(self, x_bc, y_bc):
        y_hat = self.forward(x_bc)
        return self.loss_func(y_bc, y_hat)

    def loss_pde(self, x_pde):
        x_pde.requires_grad = True
        y_hat = self.forward(x_pde)
        loss_pde = torch.tensor(0.)
        for i in range(N):
            x_i = y_hat[:, [i]]
            dx_i = autograd.grad(x_i, x_pde, torch.ones_like(x_i), create_graph=True)[0]
            f_i = (y_hat[:, [(i + 1) % N]] - y_hat[:, [i - 2]]) * y_hat[:, [i - 1]] - y_hat[:, [i]] + F
            loss_x_i = self.loss_func(dx_i, f_i)
            loss_pde = loss_pde if loss_pde > loss_x_i else loss_x_i

        return loss_pde

    def loss(self, x_bc, y_bc, x_pde):
        loss_bc = self.loss_bc(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        loss_max = loss_bc if loss_bc > loss_pde else loss_pde
        return loss_max


if "__main__" == __name__:
    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # lorenz 方程参数，方程个数及右端F项
    N = 5
    F = 8

    total_points = 300
    n_f = total_points

    x_lb = torch.tensor(0.)
    x_ub = torch.tensor(1.)

    x_test = torch.linspace(x_lb, x_ub, total_points)

    # 初值点
    x_train_bc = torch.tensor([[0.0]])
    y_train_bc = F * torch.ones((1, N))
    # 增加扰动
    y_train_bc[0][0] += 1
    # y_train_nu[0][1] += 5
    # y_train_nu[0][2] += 3
    # y_train_nu[0][3] += 8
    # y_train_nu[0][4] += 12

    # 配置点
    x_train_nf = torch.from_numpy(lhs(1, n_f)).float()
    x_train_nf = torch.vstack((x_train_bc, x_train_nf))

    layers = [1, 64, 64, N]
    PINN = FCN(layers)
    print(PINN)

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=50000,
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

        if epoch % 1000 == 0:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())

        if loss.item() < 1:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(PINN.state_dict(), 'lorenz96_pinn_01.pt')
    PINN.cpu()
    nn_predict = PINN(x_test[:, None]).detach().numpy()
    x_nn = nn_predict[:, [0]]
    y_nn = nn_predict[:, [1]]
    z_nn = nn_predict[:, [2]]

    fig, ax = plt.subplots()
    ax.plot(x_test, x_nn, color='r', label='x')
    ax.plot(x_test, y_nn, color='g', label='y')
    ax.plot(x_test, z_nn, color='b', label='z')
    ax.set_title('Lorenz 96')
    ax.set_xlabel('t', color='black')
    ax.set_ylabel('f(t)', color='black', rotation=0)
    ax.legend(loc='upper right')
    plt.savefig('./figure/lorenz96_pinn_01_2d.png')
    plt.close()
    # plt.show()

    ax = plt.axes(projection="3d")
    ax.plot(x_nn, y_nn, z_nn, color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('Lorenz 96')
    plt.savefig('./figure/lorenz96_pinn_01_3d.png')
    # plt.show()
