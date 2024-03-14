import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from pyDOE import lhs
from datetime import datetime

torch.set_default_dtype(torch.float)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class Lorenz63Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # lorenz 方程参数
        self.rho = torch.tensor(15.0).float().to(device)
        self.sigma = torch.tensor(10.0).float().to(device)
        self.beta = torch.tensor(8.0 / 3.0).float().to(device)
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

    def loss_ic(self, t_ic, u_ic):
        u_hat = self.forward(t_ic)
        return self.loss_func(u_ic, u_hat)

    def loss_pde(self, t_pde):
        t_pde.requires_grad = True
        u_hat = self.forward(t_pde)
        x_hat = u_hat[:, [0]]
        y_hat = u_hat[:, [1]]
        z_hat = u_hat[:, [2]]
        x_t = autograd.grad(x_hat, t_pde, torch.ones_like(x_hat), create_graph=True)[0]
        y_t = autograd.grad(y_hat, t_pde, torch.ones_like(y_hat), create_graph=True)[0]
        z_t = autograd.grad(z_hat, t_pde, torch.ones_like(z_hat), create_graph=True)[0]

        lhs_ = torch.hstack((x_t, y_t, z_t))
        rhs_ = torch.hstack(
            (self.sigma * (y_hat - x_hat), x_hat * (self.rho - z_hat) - y_hat, x_hat * y_hat - self.beta * z_hat))

        return self.loss_func(lhs_, rhs_)

    def loss(self, x_bc, y_bc, x_pde):
        loss_bc = self.loss_ic(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        return loss_bc + loss_pde


if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)

    layers = [1, 32, 32, 3]
    model = Lorenz63Net(layers)
    model.to(device)
    print(model)

    U = np.load('lorenz63_non_chaos.npy')

    # 初值配点
    t_train_ic = np.array([[0.]])
    u_train_ic = np.array([U[0]])

    # 物理配点
    t_train_pde = lhs(1, 200)

    # 转为张量，用于训练
    t_train_ic = torch.from_numpy(t_train_ic).float().to(device)
    u_train_ic = torch.from_numpy(u_train_ic).float().to(device)
    t_train_pde = torch.from_numpy(t_train_pde).float().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7, mode='min', factor=0.5,
                                                           patience=6000,
                                                           verbose=True)

    # 记录训练开始时间
    start_time = datetime.now()
    print("Training started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    epoch = 0
    save_threshold = 1e-3
    while True:
        epoch += 1
        loss = model.loss(t_train_ic, u_train_ic, t_train_pde)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % 1000 == 0:
            print(datetime.now(), 'epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
        if loss.item() < save_threshold:
            print(datetime.now(), 'epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'lorenz63_non_chaos_pinn_01_' + str(save_threshold) + '.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
