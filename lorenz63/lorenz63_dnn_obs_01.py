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
    """
    MSE + Input Norm + Batch Norm
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
        x_pde.requires_grad = True
        y_hat = self.forward(x_pde)
        x_nn = y_hat[:, [0]]
        y_nn = y_hat[:, [1]]
        z_nn = y_hat[:, [2]]
        x_t = autograd.grad(x_nn, x_pde, torch.ones_like(x_nn), create_graph=True)[0]
        y_t = autograd.grad(y_nn, x_pde, torch.ones_like(y_nn), create_graph=True)[0]
        z_t = autograd.grad(z_nn, x_pde, torch.ones_like(z_nn), create_graph=True)[0]
        loss_x = self.loss_func(x_t, sigma * (y_nn - x_nn))
        loss_y = self.loss_func(y_t, x_nn * (rho - z_nn) - y_nn)
        loss_z = self.loss_func(z_t, x_nn * y_nn - beta * z_nn)

        return loss_x + loss_y + loss_z

    def loss(self, x_bc, y_bc, x_pde):
        loss_bc = self.loss_bc(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        return loss_bc + loss_pde


if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)
    # float is float32
    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    layers = [1, 32, 32, 3]
    PINN = FCN(layers)
    PINN.to(device)
    print(PINN)

    # 总点数
    total_points = 601
    x_lb = torch.tensor(0.)
    x_ub = torch.tensor(3.)

    # lorenz 方程参数
    rho = torch.tensor(28.0)
    sigma = torch.tensor(10.0)
    beta = torch.tensor(8.0 / 3.0)

    x_test = torch.linspace(x_lb, x_ub, total_points)

    obs = np.load('lorenz63_10_28_8_3_0.005_600.npy')
    # 选取观测值
    idx = [i for i in range(0, len(obs), 4)]
    x_train_bc = x_test[idx].unsqueeze(1)
    y_train_bc = torch.from_numpy(obs[idx])
    print('x_train_bc :', x_train_bc)
    print('y_train_bc :', y_train_bc)

    # 配置点
    n_f = total_points
    x_train_nf = (x_lb + (x_ub - x_lb) * lhs(1, n_f)).float()

    x_train_bc = x_train_bc.float().to(device)
    y_train_bc = y_train_bc.float().to(device)
    x_train_nf = x_train_nf.float().to(device)

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=50000,
                                                           verbose=True)

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
        if loss.item() < 0.001:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    PINN = PINN.cpu()
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
    plt.savefig('./figure/lorenz63_dnn_obs_01_2d.png')
    plt.close()
    # plt.show()

    ax = plt.axes(projection="3d")
    ax.plot(x_nn, y_nn, z_nn, color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('lorenz63')
    plt.savefig('./figure/lorenz63_dnn_obs_01_3d.png')
    # plt.show()
