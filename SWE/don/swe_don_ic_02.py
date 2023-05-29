import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

"""
整个空间是100*100，PDE参数eta，
主干网的输入取左上角20*20的区域；此时标准差sigma=5，根据2sigma准则，高斯函数的中心在区域[5,15]*[5,15]
分支网的输入仍然是100*100网格上的eta值
"""


class SWEICDataset(Dataset):
    """
    初值训练集
    """

    def __init__(self, eta_train_raw, z_train_raw):
        self.eta_train_count = len(eta_train_raw)
        self.z_train_count = len(z_train_raw)
        self.length = self.eta_train_count * self.z_train_count

        eta_train = eta_train_raw[:, :20, :20].reshape(self.eta_train_count, -1)
        eta_train = np.repeat(eta_train, self.z_train_count, axis=0)
        z_train = np.tile(z_train_raw, (self.eta_train_count, 1))

        eta_label = eta_train_raw.reshape(-1, 1)
        uv_label = np.zeros((eta_label.shape[0], 2))
        label = np.hstack((uv_label, eta_label))

        self.eta_train = torch.from_numpy(eta_train).float().to(device)
        self.z_train = torch.from_numpy(z_train).float().to(device)
        self.label = torch.from_numpy(label).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.eta_train[index], self.z_train[index], self.label[index]


class SWEPhysicsDataset(Dataset):
    """
    物理信息训练集
    """

    def __init__(self, eta_train_raw, physics_train_count=1000):
        self.eta_train_count = len(eta_train_raw)
        self.z_train_count = physics_train_count
        self.length = self.eta_train_count * physics_train_count

        eta_train = eta_train_raw[:, :20, :20].reshape(self.eta_train_count, -1)
        eta_train = np.repeat(eta_train, physics_train_count, axis=0)

        z_train = np.random.uniform(low=[0, 0, 0], high=[1, 1, 1], size=(physics_train_count, 3))
        z_train = np.tile(z_train, (self.eta_train_count, 1))

        self.eta_train = torch.from_numpy(eta_train).float().to(device)
        self.z_train = torch.from_numpy(z_train).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.eta_train[index], self.z_train[index]


class SWENet(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super().__init__()
        self.H = 100
        self.g = 9.81
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.branch_linear = nn.ModuleList(
            [nn.Linear(branch_layers[i], branch_layers[i + 1]) for i in range(len(branch_layers) - 1)])
        self.trunk_linear = nn.ModuleList(
            [nn.Linear(trunk_layers[i], trunk_layers[i + 1]) for i in range(len(trunk_layers) - 1)])
        # 主干网络和分支网络进行数据融合
        self.merge_linear = nn.Linear(trunk_layers[-1], 3)

        for linear in self.branch_linear:
            nn.init.xavier_normal_(linear.weight.data)

        for linear in self.trunk_linear:
            nn.init.xavier_normal_(linear.weight.data)

        nn.init.xavier_normal_(self.merge_linear.weight.data)

    def forward(self, a_branch, a_trunk):
        for j in range(len(self.branch_linear) - 1):
            z_branch = self.branch_linear[j](a_branch)
            a_branch = self.activation(z_branch)
        z_branch = self.branch_linear[-1](a_branch)

        for i in range(len(self.trunk_linear) - 1):
            z_trunk = self.trunk_linear[i](a_trunk)
            a_trunk = self.activation(z_trunk)
        z_trunk = self.trunk_linear[-1](a_trunk)

        # 计算主干网络和分支网络输出的乘积，注意是元素级别的
        product = torch.mul(z_branch, z_trunk)
        return self.merge_linear(product)

    def loss_ic(self, ic_batch):
        eta_train, z_train, label = ic_batch
        u_hat = self.forward(eta_train, z_train)
        return self.loss_func(u_hat, label)

    def loss_physics(self, physics_batch):
        eta_train, z_train = physics_batch
        z_train.requires_grad = True
        nn_hat = self.forward(eta_train, z_train)
        u_hat = nn_hat[:, [0]]
        v_hat = nn_hat[:, [1]]
        eta_hat = nn_hat[:, [2]]

        du = autograd.grad(u_hat, z_train, torch.ones_like(u_hat), create_graph=True)[0]
        dv = autograd.grad(v_hat, z_train, torch.ones_like(v_hat), create_graph=True)[0]
        deta = autograd.grad(eta_hat, z_train, torch.ones_like(eta_hat), create_graph=True)[0]

        u_t = du[:, [2]]
        v_t = dv[:, [2]]
        eta_x = deta[:, [0]]
        eta_y = deta[:, [1]]
        eta_t = deta[:, [2]]

        q = (eta_hat + self.H) * u_hat
        s = (eta_hat + self.H) * v_hat
        dq = autograd.grad(q, z_train, torch.ones_like(q), create_graph=True)[0]
        ds = autograd.grad(s, z_train, torch.ones_like(s), create_graph=True)[0]
        q_x = dq[:, [0]]
        s_y = ds[:, [1]]

        lhs = torch.hstack((u_t, v_t, eta_t))
        rhs = torch.hstack((-self.g * eta_x, -self.g * eta_y, -q_x - s_y))

        return self.loss_func(lhs, rhs)

    def loss(self, ic_batch, physics_batch):
        loss_ic = self.loss_ic(ic_batch)
        loss_physics = self.loss_physics(physics_batch)
        return loss_ic + loss_physics


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    N_x = 101
    N_y = 101

    x_lb = 0.
    x_ub = 1.
    y_lb = 0.
    y_ub = 1.
    t_lb = 0.
    t_ub = 1.

    eta_train_count = 500
    physics_train_count = 1000

    # 构造训练数据
    mu = np.random.uniform(low=[0.05, 0.05], high=[0.15, 0.15], size=(eta_train_count, 2))
    x = np.linspace(x_lb, x_ub, N_x)
    y = np.linspace(y_lb, y_ub, N_y)
    X, Y = np.meshgrid(x, y)

    x_train_raw = X.flatten()[:, None]
    y_train_raw = Y.flatten()[:, None]
    t_train_raw = np.zeros_like(x_train_raw)
    z_train_raw = np.hstack((x_train_raw, y_train_raw, t_train_raw))

    eta_train_raw = []
    sigma = 0.02
    for m in mu:
        eta = np.exp(-((X - m[0]) ** 2 + (Y - m[1]) ** 2) / (sigma ** 2))
        eta_train_raw.append(eta)

    eta_train_raw = np.asarray(eta_train_raw)

    ic_ds = SWEICDataset(eta_train_raw, z_train_raw)
    physics_ds = SWEPhysicsDataset(eta_train_raw, physics_train_count=physics_train_count)
    print("ic_ds count:", len(ic_ds), "physics_ds count:", len(physics_ds))

    batch_size = 10000
    ic_loader = DataLoader(ic_ds, batch_size=batch_size, shuffle=True)
    physics_loader = DataLoader(physics_ds, batch_size=batch_size, shuffle=True)

    ic_iter = iter(ic_loader)
    physics_iter = iter(physics_loader)

    # 构建网络网络结构
    branch_layers = [400, 256, 128, 64, 64, 64]
    trunk_layers = [3, 64, 64, 64, 64, 64]

    model = SWENet(branch_layers, trunk_layers)
    model.to(device)
    print("model:\n", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7, mode='min', factor=0.7,
                                                           patience=5000,
                                                           verbose=True)

    # 记录训练开始时间
    start_time = datetime.now()
    print("Training started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    batch = 0
    while True:
        batch += 1

        # 从dataloader迭代器获取元素，如果到头了，就重新生成迭代器
        # 注意这里不能用itertools.cycle去实现，会导致shuffle失效，因为本质是同一个iter
        try:
            ic_batch = next(ic_iter)
        except StopIteration:
            ic_iter = iter(ic_loader)
            ic_batch = next(ic_iter)

        try:
            physics_batch = next(physics_iter)
        except StopIteration:
            physics_iter = iter(physics_loader)
            physics_batch = next(physics_iter)

        loss = model.loss(ic_batch, physics_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if batch % 100 == 0:
            print(datetime.now(), 'batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
        if loss.item() < 1E-4:
            print(datetime.now(), 'batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'swe_don_ic_02_gzz_e-6.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
