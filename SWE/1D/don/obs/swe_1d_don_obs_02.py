import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import sys
from pyDOE import lhs


class SWEObsDataset(Dataset):
    """
    观测点对应的训练集
    """

    def __init__(self, o_train_raw, y_train_raw):
        """
        根据随机函数，初始化数据集
        :param o_train_raw: 观测值
        :param y_train_raw: 观测点
        """
        self.o_train_count = len(o_train_raw)
        self.y_train_count = len(y_train_raw)
        self.length = self.o_train_count * self.y_train_count

        self.o_train = torch.from_numpy(o_train_raw).float().to(device)
        self.y_train = torch.from_numpy(y_train_raw).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        hv_idx, y_idx = divmod(index, self.y_train_count)
        hv = self.o_train[hv_idx]
        y = self.y_train[y_idx]
        label = hv[[y_idx, y_idx + self.y_train_count]]
        return hv, y, label


class SWEPhysicsDataset(Dataset):
    """
    物理信息对应的训练集
    """

    def __init__(self, o_train_raw, physics_train_count=1000):
        """
        根据随机函数，初始化数据集
        :param o_train_raw: 观测值
        :param physics_train_count:随机生成的训练点
        """
        self.o_train_count = len(o_train_raw)
        self.y_train_count = physics_train_count
        self.length = self.o_train_count * physics_train_count

        self.o_train = torch.from_numpy(o_train_raw).float().to(device)
        self.y_train = torch.from_numpy(lhs(2, physics_train_count)).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        hv_idx, y_idx = divmod(index, self.y_train_count)
        hv = self.o_train[hv_idx]
        y = self.y_train[y_idx]
        return hv, y


class SWENet(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super().__init__()
        self.g = torch.tensor(1.).float().to(device)
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.branch_linear = nn.ModuleList(
            [nn.Linear(branch_layers[i], branch_layers[i + 1]) for i in range(len(branch_layers) - 1)])
        self.trunk_linear = nn.ModuleList(
            [nn.Linear(trunk_layers[i], trunk_layers[i + 1]) for i in range(len(trunk_layers) - 1)])
        # 主干网络和分支网络进行数据融合
        self.merge_linear = nn.Linear(trunk_layers[-1], 2)

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

    def loss_obs(self, obs_batch):
        o_train, y_train, label = obs_batch
        u_hat = self.forward(o_train, y_train)
        return self.loss_func(u_hat, label)

    def loss_physics(self, physics_batch):
        o_train, y_train = physics_batch
        y_train.requires_grad = True
        nn_hat = self.forward(o_train, y_train)
        h_hat = nn_hat[:, [0]]
        v_hat = nn_hat[:, [1]]

        dh = autograd.grad(h_hat, y_train, torch.ones_like(h_hat), create_graph=True)[0]
        dv = autograd.grad(v_hat, y_train, torch.ones_like(v_hat), create_graph=True)[0]

        h_x = dh[:, [0]]
        h_t = dh[:, [1]]
        v_x = dv[:, [0]]
        v_t = dv[:, [1]]

        lhs_ = torch.hstack((h_t, h_hat * v_t + v_hat * h_t))
        rhs_ = torch.hstack(
            (-h_hat * v_x - v_hat * h_x, -2 * h_hat * v_hat * v_x - (v_hat ** 2 + self.g * h_hat) * h_x))

        return self.loss_func(lhs_, rhs_)

    def loss(self, obs_batch, physics_batch):
        loss_obs = self.loss_obs(obs_batch)
        loss_physics = self.loss_physics(physics_batch)
        return loss_obs + loss_physics


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data = np.load('swe_1d_don_obs_01.npz')
    o_train = train_data['o_train']
    y_train = train_data['y_train']

    # 构建网络网络结构
    branch_layers = [o_train.shape[1], 128, 64, 64, 64, 64]
    trunk_layers = [y_train.shape[1], 64, 64, 64, 64, 64]

    model = SWENet(branch_layers, trunk_layers)
    model.to(device)
    print("model:\n", model)

    obs_ds = SWEObsDataset(o_train, y_train)
    physics_ds = SWEPhysicsDataset(o_train, 1000)
    print("obs_ds count:", len(obs_ds), "physics_ds count:", len(physics_ds))

    batch_size = 10000
    obs_loader = DataLoader(obs_ds, batch_size=batch_size, shuffle=True)
    physics_loader = DataLoader(physics_ds, batch_size=batch_size, shuffle=True)

    obs_iter = iter(obs_loader)
    physics_iter = iter(physics_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-8, mode='min', factor=0.5,
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
            obs_batch = next(obs_iter)
        except StopIteration:
            obs_iter = iter(obs_loader)
            obs_batch = next(obs_iter)

        try:
            physics_batch = next(physics_iter)
        except StopIteration:
            physics_iter = iter(physics_loader)
            physics_batch = next(physics_iter)

        loss = model.loss(obs_batch, physics_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if batch % 100 == 0:
            print('batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
        if loss.item() < 1E-5:
            print('batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'swe_1d_don_obs_02.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
