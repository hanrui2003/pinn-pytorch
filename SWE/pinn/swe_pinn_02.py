import torch
import numpy as np
from pyDOE import lhs
import torch.nn as nn
import torch.autograd as autograd
from datetime import datetime


class SWENet(nn.Module):
    def __init__(self, layers, lb, ub, interval_len):
        super().__init__()
        self.H = torch.tensor(100).float().to(device)
        self.g = torch.tensor(9.81).float().to(device)
        self.lb = torch.from_numpy(lb).float().to(device)
        self.ub = torch.from_numpy(ub).float().to(device)
        self.interval_len = torch.from_numpy(interval_len).float().to(device)
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')

        self.linears = nn.ModuleList([nn.Linear(layers[j], layers[j + 1]) for j in range(len(layers) - 1)])

        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def forward(self, a):
        a = (a - self.lb) / self.interval_len
        for j in range(len(self.linears) - 1):
            z = self.linears[j](a)
            a = self.activation(z)
        return self.linears[-1](a)

    def loss_ic(self, z_train, label):
        hat = self.forward(z_train)
        return self.loss_func(label, hat)

    def loss_pde(self, z_pde):
        z_pde.requires_grad = True
        nn_hat = self.forward(z_pde)
        u_hat = nn_hat[:, [0]]
        v_hat = nn_hat[:, [1]]
        eta_hat = nn_hat[:, [2]]

        du = autograd.grad(u_hat, z_pde, torch.ones_like(u_hat), create_graph=True)[0]
        dv = autograd.grad(v_hat, z_pde, torch.ones_like(v_hat), create_graph=True)[0]
        deta = autograd.grad(eta_hat, z_pde, torch.ones_like(eta_hat), create_graph=True)[0]

        u_x = du[:, [0]]
        u_t = du[:, [2]]
        v_y = dv[:, [1]]
        v_t = dv[:, [2]]
        eta_x = deta[:, [0]]
        eta_y = deta[:, [1]]
        eta_t = deta[:, [2]]

        lhs_ = torch.hstack((u_t, v_t, eta_t))
        rhs_ = torch.hstack(
            (-self.g * eta_x, -self.g * eta_y, -u_hat * eta_x - v_hat * eta_y - (eta_hat + self.H) * (u_x + v_y)))

        return self.loss_func(lhs_, rhs_)

    def loss(self, z_train_ic, label, z_train_nf):
        loss_bc = self.loss_ic(z_train_ic, label)
        loss_pde = self.loss_pde(z_train_nf)
        return loss_bc + loss_pde


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 空间网格设置
    N_x = 101
    N_y = 101

    x_len = 1E+6
    y_len = 1E+6
    t_len = 1E+5
    interval_len = np.array([x_len, y_len, t_len])

    lb = np.array([-x_len / 2, -y_len / 2, 0])
    ub = np.array([x_len / 2, y_len / 2, t_len])

    x = np.linspace(-x_len / 2, x_len / 2, N_x)
    y = np.linspace(-y_len / 2, y_len / 2, N_y)
    X, Y = np.meshgrid(x, y)

    x_train_ic = X.reshape(-1, 1)
    y_train_ic = Y.reshape(-1, 1)
    t_train_ic = np.zeros((N_x * N_y, 1))
    z_train_ic = np.hstack((x_train_ic, y_train_ic, t_train_ic))

    # 高斯函数设置
    mu = [x_len / 2.7, y_len / 4]
    sigma = 0.05E+6
    eta_label = np.exp(-((X - mu[0]) ** 2 + (Y - mu[1]) ** 2) / (2 * sigma ** 2))
    eta_label = eta_label.reshape(-1, 1)
    uv_label = np.zeros((N_x * N_y, 2))
    label = np.hstack((uv_label, eta_label))

    z_train_nf = lb + interval_len * lhs(3, 100000)

    z_train_ic = torch.from_numpy(z_train_ic).float().to(device)
    label = torch.from_numpy(label).float().to(device)
    z_train_nf = torch.from_numpy(z_train_nf).float().to(device)

    layers = [3, 64, 64, 64, 64, 64, 64, 3]
    model = SWENet(layers, lb, ub, interval_len)
    model.to(device)
    print("model:\n", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7, mode='min', factor=0.5,
                                                           patience=5000,
                                                           verbose=True)

    # 记录训练开始时间
    start_time = datetime.now()
    print("Training started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    epoch = 0
    while True:
        epoch += 1
        loss = model.loss(z_train_ic, label, z_train_nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % 100 == 0:
            print(datetime.now(), 'epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
        if loss.item() < 1E-6:
            print(datetime.now(), 'epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'swe_pinn_02.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
