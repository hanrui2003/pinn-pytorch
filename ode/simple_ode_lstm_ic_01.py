import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs


class FCN(nn.Module):
    def __init__(self, x_lb, x_ub):
        super().__init__()
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.loss_func = nn.MSELoss(reduction='mean')
        self.lstm = nn.LSTM(input_size=1, hidden_size=16, num_layers=1, batch_first=True)
        self.linear = nn.Linear(in_features=16, out_features=1)
        nn.init.xavier_normal_(self.linear.weight.data)
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, x):
        # 正规化
        x = (x - self.x_lb) / (self.x_ub - self.x_lb)
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x

    def loss_bc(self, x_bc, y_bc):
        y_hat = self.forward(x_bc)
        return self.loss_func(y_bc, y_hat)

    def loss_pde(self, x_pde):
        x_pde.requires_grad = True
        y_hat = self.forward(x_pde)
        # x_nn = y_hat[:, [0]]
        # y_nn = y_hat[:, [1]]
        # z_nn = y_hat[:, [2]]
        # x_t = autograd.grad(x_nn, x_pde, torch.ones_like(x_nn), create_graph=True)[0]
        # y_t = autograd.grad(y_nn, x_pde, torch.ones_like(y_nn), create_graph=True)[0]
        # z_t = autograd.grad(z_nn, x_pde, torch.ones_like(z_nn), create_graph=True)[0]
        # loss_x = self.loss_func(x_t, sigma * (y_nn - x_nn))
        # loss_y = self.loss_func(y_t, x_nn * (rho - z_nn) - y_nn)
        # loss_z = self.loss_func(z_t, x_nn * y_nn - beta * z_nn)
        u_t = autograd.grad(y_hat, x_pde, torch.ones_like(y_hat), create_graph=True)[0]
        return self.loss_func(u_t, -torch.sin(x_pde))

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
    # 区间总数
    total_interval = 1200
    # 总点数
    total_points = total_interval + 1
    # 区间长度
    h = 0.005
    x_lb = torch.tensor(0.)
    x_ub = torch.tensor(total_interval * h)

    PINN = FCN(x_lb, x_ub)
    PINN.to(device)
    print(PINN)

    x_test = torch.linspace(x_lb, x_ub, total_points)

    # 初值
    x_train_bc = torch.tensor([[0.]])
    y_train_bc = torch.tensor([[1.]])

    # 配置点
    x_train_nf = x_test.unsqueeze(1)

    x_train_bc = x_train_bc.float().to(device)
    y_train_bc = y_train_bc.float().to(device)
    x_train_nf = x_train_nf.float().to(device)

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=40000,
                                                           verbose=True)
    epoch = 0
    while True:
        epoch += 1
        with torch.backends.cudnn.flags(enabled=False):
            loss = PINN.loss_pde(x_train_bc, y_train_bc, x_train_nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 100 == 0:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())

        if loss.item() < 0.001:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    PINN.cpu()
    nn_predict = PINN(x_test[:, None]).detach().numpy()
    # x_nn = nn_predict[:, [0]]
    # y_nn = nn_predict[:, [1]]
    # z_nn = nn_predict[:, [2]]

    fig, ax = plt.subplots()
    ax.plot(x_test, nn_predict, color='r', label='x')
    # ax.plot(x_test, y_nn, color='g', label='y')
    # ax.plot(x_test, z_nn, color='b', label='z')
    ax.set_title('$x^2$')
    ax.set_xlabel('t', color='black')
    ax.set_ylabel('f(t)', color='black', rotation=0)
    ax.legend(loc='upper right')
    plt.savefig('./figure/simple_ode_lstm_ic_01.png')
    # plt.show()
