import torch
import numpy as np
from pyDOE import lhs
import torch.nn as nn
import torch.autograd as autograd
from datetime import datetime


class SWENet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.g = torch.tensor(1.).float().to(device)
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

    def loss_ic(self, y_train_ic, label_ic):
        hat = self.forward(y_train_ic)
        return self.loss_func(label_ic, hat)

    def loss_bc(self, y_train_bc, v_label_bc):
        hat = self.forward(y_train_bc)
        v_hat = hat[:, [1]]
        return self.loss_func(v_label_bc, v_hat)

    def loss_pde(self, y_pde):
        y_pde.requires_grad = True
        nn_hat = self.forward(y_pde)
        h_hat = nn_hat[:, [0]]
        v_hat = nn_hat[:, [1]]

        dh = autograd.grad(h_hat, y_pde, torch.ones_like(h_hat), create_graph=True)[0]
        dv = autograd.grad(v_hat, y_pde, torch.ones_like(v_hat), create_graph=True)[0]

        h_x = dh[:, [0]]
        h_t = dh[:, [1]]
        v_x = dv[:, [0]]
        v_t = dv[:, [1]]

        lhs_ = torch.hstack((h_t, h_hat * v_t + v_hat * h_t))
        rhs_ = torch.hstack(
            (-h_hat * v_x - v_hat * h_x, -2 * h_hat * v_hat * v_x - (v_hat ** 2 + self.g * h_hat) * h_x))

        return self.loss_func(lhs_, rhs_)

    def loss(self, y_train_ic, label_ic, y_train_bc, v_label_bc, y_train_nf):
        loss_ic = self.loss_ic(y_train_ic, label_ic)
        loss_bc = self.loss_bc(y_train_bc, v_label_bc)
        loss_pde = self.loss_pde(y_train_nf)
        return loss_ic + loss_bc + loss_pde


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 初值条件
    x_train_ic = np.linspace(0, 1, 101)[:, None]
    t_train_ic = np.zeros_like(x_train_ic)
    y_train_ic = np.hstack((x_train_ic, t_train_ic))

    h_label_ic_left = 0.2 * np.ones((21, 1))
    h_label_ic_right = 0.1 * np.ones((80, 1))
    h_label_ic = np.vstack((h_label_ic_left, h_label_ic_right))
    v_label_ic = np.zeros_like(h_label_ic)
    label_ic = np.hstack((h_label_ic, v_label_ic))

    # 边值条件
    t_train_bc = np.linspace(0, 1, 101)[:, None]
    x_train_bc1 = np.zeros_like(t_train_bc)
    x_train_bc2 = np.ones_like(t_train_bc)
    y_train_bc1 = np.hstack((x_train_bc1, t_train_bc))
    y_train_bc2 = np.hstack((x_train_bc2, t_train_bc))
    y_train_bc = np.vstack((y_train_bc1, y_train_bc2))

    v_label_bc = np.zeros((y_train_bc.shape[0], 1))

    lb = y_train_bc[0]
    ub = y_train_bc[-1]
    # 配置点
    y_train_nf = lb + (ub - lb) * lhs(2, 10000)

    y_train_ic = torch.from_numpy(y_train_ic).float().to(device)
    label_ic = torch.from_numpy(label_ic).float().to(device)
    y_train_bc = torch.from_numpy(y_train_bc).float().to(device)
    v_label_bc = torch.from_numpy(v_label_bc).float().to(device)
    y_train_nf = torch.from_numpy(y_train_nf).float().to(device)

    layers = [2, 64, 64, 64, 64, 64, 64, 2]
    model = SWENet(layers)
    model.to(device)
    print("model:\n", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7, mode='min', factor=0.5,
                                                           patience=5000,
                                                           verbose=True)

    # 记录训练开始时间
    start_time = datetime.now()
    print("Training started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    epoch = 0
    while True:
        epoch += 1
        loss = model.loss(y_train_ic, label_ic, y_train_bc, v_label_bc, y_train_nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % 100 == 0:
            print(datetime.now(), 'epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
        if loss.item() < 1E-6:
            print(datetime.now(), 'epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'swe_1d_pinn_03.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
