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


class ConvDiffNet(nn.Module):
    def __init__(self, layers):
        super().__init__()
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

    # 不妨选取第一个样本序列，时间t=[0,1]的数据，即101*202
    sample_data = np.load('swe_1d_rbf_sample_1.npy')
    train_data = sample_data[0][:101, :]

    # 初值条件
    x_train_ic = np.linspace(0, 1, 101)[:, None]
    t_train_ic = np.zeros_like(x_train_ic)
    y_train_ic = np.hstack((x_train_ic, t_train_ic))

    h_label_ic = train_data[0][:101][:, None]
    v_label_ic = train_data[0][101:][:, None]
    label_ic = np.hstack((h_label_ic, v_label_ic))

    # 边值条件
    t_train_bc = np.linspace(0, 1, 101)[:, None]
    x_train_bc1 = np.zeros_like(t_train_bc)
    x_train_bc2 = np.ones_like(t_train_bc)
    y_train_bc1 = np.hstack((x_train_bc1, t_train_bc))
    y_train_bc2 = np.hstack((x_train_bc2, t_train_bc))
    y_train_bc = np.vstack((y_train_bc1, y_train_bc2))

    v_label_bc = np.zeros((y_train_bc.shape[0], 1))

    # 观测条件
    t_train_obs = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])[:, None]
    x_train_obs = 0.5 * np.ones_like(t_train_obs)
    y_train_obs = np.hstack((x_train_obs, t_train_obs))

    delta_index = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    h_label_obs = train_data[delta_index, 50][:, None]
    v_label_obs = train_data[delta_index, 151][:, None]
    label_obs = np.hstack((h_label_obs, v_label_obs))

    # 把初值和观测数据合并，因为本质都是带标签的数据
    y_train_ic_obs = np.vstack((y_train_ic, y_train_obs))
    label_ic_obs = np.vstack((label_ic, label_obs))

    # 配置点
    y_train_nf = lhs(2, 10000)

    # 把训练数据转为tensor
    y_train_ic_obs = torch.from_numpy(y_train_ic_obs).float().to(device)
    label_ic_obs = torch.from_numpy(label_ic_obs).float().to(device)
    y_train_bc = torch.from_numpy(y_train_bc).float().to(device)
    v_label_bc = torch.from_numpy(v_label_bc).float().to(device)
    y_train_nf = torch.from_numpy(y_train_nf).float().to(device)

    layers = [2, 32, 32, 32, 32, 32, 32, 2]
    model = ConvDiffNet(layers)
    model.to(device)
    print("model:\n", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7, mode='min', factor=0.5,
                                                           patience=6000,
                                                           verbose=True)

    # 记录训练开始时间
    start_time = datetime.now()
    print("Training started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    epoch = 0
    while True:
        epoch += 1
        loss = model.loss(y_train_ic_obs, label_ic_obs, y_train_bc, v_label_bc, y_train_nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % 100 == 0:
            print(datetime.now(), 'epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
        if loss.item() < 5E-6:
            print(datetime.now(), 'epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'swe_1d_pinn_obs_01.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
