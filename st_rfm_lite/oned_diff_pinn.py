import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import sys
from pyDOE import lhs

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

Pi = np.pi
nu = 0.01

X_min = 0.0
X_max = 5.0
T_min = 0.0
T_max = 1.0

Nx = 5
# t维度划分的区间数
Nt = 2
# 每个局部局域的特征函数数量
M = 150
# x维度每个区间的配点数，Qx+1
Qx = 30
# t维度每个区间的配点数，Qt+1
Qt = 30


def u_real(x, t):
    u0 = 2 * np.cos(Pi * x + Pi / 5) + 3 / 2 * np.cos(2 * Pi * x - 3 * Pi / 5)
    u1 = 2 * np.cos(Pi * t + Pi / 5) + 3 / 2 * np.cos(2 * Pi * t - 3 * Pi / 5)

    u = u0 * u1

    return u


def f_real(x, t):
    u0 = 2 * np.cos(Pi * x + Pi / 5) + 3 / 2 * np.cos(2 * Pi * x - 3 * Pi / 5)
    u1 = 2 * np.cos(Pi * t + Pi / 5) + 3 / 2 * np.cos(2 * Pi * t - 3 * Pi / 5)

    u1t = -2 * Pi * np.sin(Pi * t + Pi / 5) - 3 * Pi * np.sin(2 * Pi * t - 3 * Pi / 5)
    ut = u0 * u1t

    u0xx = -2 * Pi * Pi * np.cos(Pi * x + Pi / 5) - 6 * Pi * Pi * np.cos(2 * Pi * x - 3 * Pi / 5)
    uxx = u0xx * u1

    f = ut - nu * uxx

    return f


class DiffNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.nu = torch.tensor(0.01).float().to(device)
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')

        self.linears = nn.ModuleList([nn.Linear(layers[j], layers[j + 1]) for j in range(len(layers) - 1)])

        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def forward(self, a):
        for j in range(len(self.linears) - 1):
            z = self.linears[j](a)
            a = self.activation(z)
        return self.linears[-1](a)

    def loss_label(self, y, label):
        """
        带标签的点
        """
        hat = self.forward(y)
        return self.loss_func(label, hat)

    def loss_pde(self, y_pde, label_pde):
        """
        PDE的点
        """
        y_pde.requires_grad = True
        u_hat = self.forward(y_pde)

        u_x_t = autograd.grad(u_hat, y_pde, torch.ones_like(u_hat), create_graph=True)[0]
        u_x = u_x_t[:, [0]]
        u_t = u_x_t[:, [1]]

        u_xx_xt = autograd.grad(u_x, y_pde, torch.ones_like(u_x), create_graph=True)[0]
        u_xx = u_xx_xt[:, [0]]

        return self.loss_func(u_t - self.nu * u_xx, label_pde)

    def loss(self, y_ic_bc, label_ic_bc, y_pde, label_pde):
        loss_ic_bc = self.loss_label(y_ic_bc, label_ic_bc)
        loss_pde = self.loss_pde(y_pde, label_pde)
        return loss_ic_bc + loss_pde


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    x = np.linspace(X_min, X_max, Nx * Qx + 1)
    t = np.linspace(T_min, T_max, Nt * Qt + 1)
    X, T = np.meshgrid(x, t)
    # 用于计算pde损失的点
    y_train_pde = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    y_label_pde = f_real(y_train_pde[:, [0]], y_train_pde[:, [1]])

    # 带标签值的点，即有真解的点
    ic_points = np.hstack((X[0][:, None], T[0][:, None]))
    left_bc_points = np.hstack((X[:, 0][:, None], T[:, 0][:, None]))
    right_bc_points = np.hstack((X[:, -1][:, None], T[:, -1][:, None]))
    bc_points = np.vstack((left_bc_points, right_bc_points))
    y_train_ic_bc = np.vstack((ic_points, bc_points))
    y_label_ic_bc = u_real(y_train_ic_bc[:, [0]], y_train_ic_bc[:, [1]])

    # 把训练数据转为tensor
    y_train_ic_bc = torch.from_numpy(y_train_ic_bc).float().to(device)
    y_label_ic_bc = torch.from_numpy(y_label_ic_bc).float().to(device)
    y_train_pde = torch.from_numpy(y_train_pde).float().to(device)
    y_label_pde = torch.from_numpy(y_label_pde).float().to(device)

    layers = [2, 32, 32, 32, 32, 1]
    model = DiffNet(layers)
    model.to(device)
    print("model:\n", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2000, gamma=0.5)

    # 记录训练开始时间
    start_time = datetime.now()
    print("Training started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    for epoch in range(20000):
        loss = model.loss(y_train_ic_bc, y_label_ic_bc, y_train_pde, y_label_pde)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(datetime.now(), 'epoch :', epoch + 1, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())

    torch.save(model, 'oned_diff_pinn_01.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
