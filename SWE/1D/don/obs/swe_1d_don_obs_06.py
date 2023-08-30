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
    初值训练集
    """

    def __init__(self, o_train):
        x_train_ic = np.linspace(0, 1, 101)[:, None]
        t_train_ic = np.zeros_like(x_train_ic)
        y_train_ic = np.hstack((x_train_ic, t_train_ic))

        t_train_obs = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])[:, None]
        x_train_obs = 0.5 * np.ones_like(t_train_obs)
        y_train_obs = np.hstack((x_train_obs, t_train_obs))

        y_train = np.vstack((y_train_ic, y_train_obs))

        o_train_count = len(o_train)
        y_train_count = len(y_train)
        self.length = o_train_count * y_train_count

        # 最终神经网络的输入
        o_train_nn = np.repeat(o_train, y_train_count, axis=0)
        y_train_nn = np.tile(y_train, (o_train_count, 1))

        x_len = o_train.shape[1] // 2
        h_label = o_train[:, :x_len].reshape(-1, 1)
        v_label = o_train[:, x_len:].reshape(-1, 1)
        label = np.hstack((h_label, v_label))
        print('obs dataset size (mb):',
              (sys.getsizeof(o_train_nn) + sys.getsizeof(y_train_nn) + sys.getsizeof(label)) / 1024 / 1024)

        self.o_train_nn = torch.from_numpy(o_train_nn).float().to(device)
        self.y_train_nn = torch.from_numpy(y_train_nn).float().to(device)
        self.label = torch.from_numpy(label).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.o_train_nn[index], self.y_train_nn[index], self.label[index]


class SWEBCDataset(Dataset):
    """
    初值训练集
    """

    def __init__(self, o_train):
        t_train_bc = np.linspace(0, 1, 101)[:, None]
        x_train_bc1 = np.zeros_like(t_train_bc)
        x_train_bc2 = np.ones_like(t_train_bc)
        y_train_bc1 = np.hstack((x_train_bc1, t_train_bc))
        y_train_bc2 = np.hstack((x_train_bc2, t_train_bc))
        y_train_bc = np.vstack((y_train_bc1, y_train_bc2))

        o_train_count = len(o_train)
        y_train_count = len(y_train_bc)
        self.length = o_train_count * y_train_count

        o_train_nn = np.repeat(o_train, y_train_count, axis=0)
        y_train_nn = np.tile(y_train_bc, (o_train_count, 1))

        v_label = np.zeros((y_train_nn.shape[0], 1))

        print('bc dataset size (mb):',
              (sys.getsizeof(o_train_nn) + sys.getsizeof(y_train_nn) + sys.getsizeof(v_label)) / 1024 / 1024)

        self.o_train_nn = torch.from_numpy(o_train_nn).float().to(device)
        self.y_train_nn = torch.from_numpy(y_train_nn).float().to(device)
        self.v_label = torch.from_numpy(v_label).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.o_train_nn[index], self.y_train_nn[index], self.v_label[index]


class SWEPhysicsDataset(Dataset):
    """
    物理信息训练集
    """

    def __init__(self, o_train, physics_train_count=5000):
        o_train_count = len(o_train)
        self.length = o_train_count * physics_train_count

        o_train_nn = np.repeat(o_train, physics_train_count, axis=0)

        y_physics = lhs(2, physics_train_count)
        y_train_nn = np.tile(y_physics, (o_train_count, 1))

        print('physics dataset size (mb):', (sys.getsizeof(o_train_nn) + sys.getsizeof(y_train_nn)) / 1024 / 1024)

        self.o_train_nn = torch.from_numpy(o_train_nn).float().to(device)
        self.y_train_nn = torch.from_numpy(y_train_nn).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.o_train_nn[index], self.y_train_nn[index]


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

    def loss_bc(self, bc_batch):
        o_train, y_train, v_label = bc_batch
        u_hat = self.forward(o_train, y_train)
        v_hat = u_hat[:, [1]]
        return self.loss_func(v_hat, v_label)

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

    def loss(self, obs_batch, bc_batch, physics_batch):
        loss_obs = self.loss_obs(obs_batch)
        loss_bc = self.loss_bc(bc_batch)
        loss_physics = self.loss_physics(physics_batch)
        return loss_obs + loss_bc + loss_physics


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data = np.load('swe_1d_rbf_sample_1.npy')

    # 观测位置选择x=0.5，对应的高度和速度的索引为
    # 遍历每个样本序列
    delta_index = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    obs_data = np.empty((0, 222))
    for seq in train_data:
        # 从每个样本序列中前500个状态中随机选择n个作为初值
        random_idx = np.random.choice(500, 1, False)
        random_ics = seq[random_idx, :]
        # 对于每个初值，不妨选取中间的点作为观测点，每隔0.1取一个观测值，对应的索引间隔就是10
        for idx, ic in enumerate(random_ics):
            obs_index = random_idx[idx] + delta_index
            obs_h = seq[obs_index, 50]
            obs_v = seq[obs_index, 151]
            obs = np.concatenate((ic[:101], obs_h, ic[101:], obs_v))
            obs_data = np.vstack((obs_data, obs))

    # 构建网络网络结构
    branch_layers = [222, 200, 100, 50, 50, 50]
    trunk_layers = [2, 50, 50, 50, 50, 50]

    model = SWENet(branch_layers, trunk_layers)
    model.to(device)
    print("model:\n", model)

    obs_ds = SWEObsDataset(obs_data)
    bc_ds = SWEBCDataset(obs_data)
    physics_ds = SWEPhysicsDataset(obs_data)
    print("obs_ds count:", len(obs_ds), "bc_ds count:", len(bc_ds), "physics_ds count:", len(physics_ds))

    batch_size = 10000
    obs_loader = DataLoader(obs_ds, batch_size=batch_size, shuffle=True)
    bc_loader = DataLoader(bc_ds, batch_size=batch_size, shuffle=True)
    physics_loader = DataLoader(physics_ds, batch_size=batch_size, shuffle=True)

    obs_iter = iter(obs_loader)
    bc_iter = iter(bc_loader)
    physics_iter = iter(physics_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-7, mode='min', factor=0.5,
                                                           patience=6000,
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
            bc_batch = next(bc_iter)
        except StopIteration:
            bc_iter = iter(bc_loader)
            bc_batch = next(bc_iter)

        try:
            physics_batch = next(physics_iter)
        except StopIteration:
            physics_iter = iter(physics_loader)
            physics_batch = next(physics_iter)

        loss = model.loss(obs_batch, bc_batch, physics_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if batch % 100 == 0:
            print(datetime.now(), 'batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
        if loss.item() < 1E-5:
            print(datetime.now(), 'batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'swe_1d_don_obs_06.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
