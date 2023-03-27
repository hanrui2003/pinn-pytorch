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

    def __init__(self, layers, x_lb, x_ub):
        super().__init__()
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[j], layers[j + 1]) for j in range(len(layers) - 1)])
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        # 正规化
        a = (x - self.x_lb) / (self.x_ub - self.x_lb)
        for j in range(len(self.linears) - 1):
            z = self.linears[j](a)
            a = self.activation(z)
        return self.linears[-1](a)

    def loss_bc(self, x_bc, y_bc):
        y_hat = self.forward(x_bc)
        return self.loss_func(y_bc, y_hat[:, [0, 2]])

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

    # 区间总数
    total_interval = 600
    # 总点数
    total_points = total_interval + 1
    # 区间长度
    h = 0.005
    x_lb = torch.tensor(0.)
    x_ub = torch.tensor(total_interval * h)

    # lorenz 方程参数
    rho = torch.tensor(28.0)
    sigma = torch.tensor(10.0)
    beta = torch.tensor(8.0 / 3.0)

    layers = [1, 32, 32, 32, 32, 32, 32, 3]
    PINN = FCN(layers, x_lb, x_ub)
    PINN.to(device)
    print(PINN)

    x_test = torch.linspace(x_lb, x_ub, total_points)

    obs = np.loadtxt('./data/iobsdisturb.txt')
    # 选取观测值
    x_train_bc = torch.from_numpy(obs[:, [0]])
    # 第1列为x的观测，第2列为z的观测，y没有观测。
    y_train_bc = torch.from_numpy(obs[:, [1, 2]])
    print('x_train_bc :', x_train_bc)
    print('y_train_bc :', y_train_bc)

    truth = np.loadtxt('./data/itruthdisturb.txt')
    t_truth = truth[:, 0]
    x_truth = truth[:, 1]
    y_truth = truth[:, 2]
    z_truth = truth[:, 3]

    # 配置点
    n_f = 100
    x_train_nf = (x_lb + (x_ub - x_lb) * lhs(1, n_f)).float()
    x_train_nf = torch.vstack((x_train_bc, x_train_nf))
    print('x_train_nf :', x_train_nf)

    x_train_bc = x_train_bc.float().to(device)
    y_train_bc = y_train_bc.float().to(device)
    x_train_nf = x_train_nf.float().to(device)

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-8, mode='min', factor=0.5,
                                                           patience=40000,
                                                           verbose=True)

    epoch = 0
    while True:
        epoch += 1
        loss = PINN.loss(x_train_bc, y_train_bc, x_train_nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        lr = optimizer.param_groups[0]['lr']
        if epoch % 1000 == 0:
            print('epoch :', epoch, 'lr :', lr, 'loss :', loss.item())
        if lr < 1e-7 or loss.item() < 0.1:
            print('epoch :', epoch, 'lr :', lr, 'loss :', loss.item())
            break

    torch.save(PINN, 'lorenz63_dnn_obs_03.pt')
    PINN = PINN.cpu()
    nn_predict = PINN(x_test[:, None]).detach().numpy()
    x_nn = nn_predict[:, [0]]
    y_nn = nn_predict[:, [1]]
    z_nn = nn_predict[:, [2]]

    fig, ax = plt.subplots()
    ax.plot(x_test, x_nn, 'b-', linewidth=2, label='x-predict')
    ax.plot(x_test, y_nn, 'g-', linewidth=2, label='y-predict')
    ax.plot(x_test, z_nn, 'k-', linewidth=2, label='z-predict')
    ax.plot(t_truth, x_truth, 'r--', linewidth=2, label='x-truth')
    ax.plot(t_truth, y_truth, 'c--', linewidth=2, label='y-truth')
    ax.plot(t_truth, z_truth, 'm--', linewidth=2, label='z-truth')
    ax.set_title('Lorenz63')
    ax.set_xlabel('t', color='black')
    ax.set_ylabel('f(t)', color='black', rotation=0)
    ax.legend(loc='upper right')
    plt.savefig('./images/lorenz63_dnn_obs_03_2d.png')
    plt.close()
    # plt.show()

    ax = plt.axes(projection="3d")
    ax.plot(x_nn, y_nn, z_nn, color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('lorenz63')
    plt.savefig('./images/lorenz63_dnn_obs_03_3d.png')
    # plt.show()
