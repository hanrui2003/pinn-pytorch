import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs


# y=sin(t+theta) 参数是theta in [0,50] ，是函数向左的偏移量
class PIDeepONet(nn.Module):
    def __init__(self, trunk_layers, branch_layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.trunk_linear = nn.ModuleList(
            [nn.Linear(trunk_layers[i], trunk_layers[i + 1]) for i in range(len(trunk_layers) - 1)])
        self.branch_linear = nn.ModuleList(
            [nn.Linear(branch_layers[i], branch_layers[i + 1]) for i in range(len(branch_layers) - 1)])

    def forward(self, a_trunk, a_branch):
        for i in range(len(self.trunk_linear) - 1):
            z_trunk = self.trunk_linear[i](a_trunk)
            a_trunk = self.activation(z_trunk)
        z_trunk = self.trunk_linear[-1](a_trunk)

        for j in range(len(self.branch_linear) - 1):
            z_branch = self.branch_linear[j](a_branch)
            a_branch = self.activation(z_branch)
        z_branch = self.branch_linear[-1](a_branch)
        # pytorch 的内积操作只适用于两个一维向量，而神经网络的训练是批量计算的；
        # 所以这里用一个变通的方式，批量计算内积
        dot = torch.sum(torch.mul(z_trunk, z_branch), dim=1, keepdim=True)
        return dot

    def loss(self, t_train, theta):
        t_train.requires_grad = True
        y_hat = self.forward(t_train, theta)

        # 计算物理损失
        y_t = autograd.grad(y_hat, t_train, torch.ones_like(y_hat), create_graph=True)[0]
        loss_physics = self.loss_func(y_t, torch.cos(t_train + theta))

        # 计算初值损失
        # 初值点对应index
        ic_idx = [i for i in range(0, len(t_train), n_t)]
        y_hat_ic = y_hat[ic_idx]
        y_label_ic = torch.sin(theta[ic_idx])
        loss_ic = self.loss_func(y_hat_ic, y_label_ic)

        return loss_physics + loss_ic


if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)
    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 生成初值数据，均匀采样
    n_theta = 500
    theta = torch.rand(n_theta) * 10
    print("theta", theta[0:100])

    # 训练点（配置点）
    n_t = 50
    t_train = torch.linspace(0, 1, n_t)

    theta = theta.unsqueeze(1).tile((1, n_t)).reshape((-1, 1))
    t_train = t_train.tile(n_theta).reshape((-1, 1))

    trunk_layers = [1, 20, 20]
    branch_layers = [1, 20, 20]

    PIDON = PIDeepONet(trunk_layers, branch_layers)

    optimizer = torch.optim.Adam(PIDON.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=10000,
                                                           verbose=True)

    epoch = 0
    while True:
        epoch += 1
        loss = PIDON.loss(t_train, theta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % 1000 == 0:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
        if loss.item() < 0.001:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(PIDON, 'sin_don_02.pt')
