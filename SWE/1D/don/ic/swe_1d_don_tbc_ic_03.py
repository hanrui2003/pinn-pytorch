import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from pyDOE import lhs
import sys


class SWEICDataset(Dataset):
    """
    初值训练集
    """

    def __init__(self, ic_train_raw, y_train_raw):
        ic_train_count = len(ic_train_raw)
        y_train_count = len(y_train_raw)
        self.length = ic_train_count * y_train_count

        ic_train = np.repeat(ic_train_raw, y_train_count, axis=0)
        y_train = np.tile(y_train_raw, (ic_train_count, 1))

        # 拼接输入
        z_train = np.hstack((y_train, ic_train))

        x_len = ic_train_raw.shape[1] // 2
        h_label = ic_train_raw[:, :x_len].reshape(-1, 1)
        v_label = ic_train_raw[:, x_len:].reshape(-1, 1)
        label = np.hstack((h_label, v_label))
        print('ic dataset size (mb):', (sys.getsizeof(z_train) + sys.getsizeof(label)) / 1024 / 1024)

        self.z_train = torch.from_numpy(z_train).float().to(device)
        self.label = torch.from_numpy(label).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.z_train[index], self.label[index]


class SWEBCDataset(Dataset):
    """
    初值训练集
    """

    def __init__(self, ic_train_raw, y_train_bc):
        ic_train_count = len(ic_train_raw)
        y_train_count = len(y_train_bc)
        self.length = ic_train_count * y_train_count

        ic_train = np.repeat(ic_train_raw, y_train_count, axis=0)
        y_train = np.tile(y_train_bc, (ic_train_count, 1))

        # 拼接输入
        z_train = np.hstack((y_train, ic_train))

        v_label = np.zeros((y_train.shape[0], 1))

        print('bc dataset size (mb):', (sys.getsizeof(z_train) + sys.getsizeof(v_label)) / 1024 / 1024)

        self.z_train = torch.from_numpy(z_train).float().to(device)
        self.v_label = torch.from_numpy(v_label).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.z_train[index], self.v_label[index]


class SWEPhysicsDataset(Dataset):
    """
    物理信息训练集
    """

    def __init__(self, ic_train_raw, physics_train_count=10000):
        ic_train_count = len(ic_train_raw)
        self.length = ic_train_count * physics_train_count

        ic_train = np.repeat(ic_train_raw, physics_train_count, axis=0)

        y_physics = lhs(2, physics_train_count)
        y_train = np.tile(y_physics, (ic_train_count, 1))

        # 拼接输入
        z_train = np.hstack((y_train, ic_train))

        print('physics dataset size (mb):', sys.getsizeof(z_train) / 1024 / 1024)

        self.z_train = torch.from_numpy(z_train).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.z_train[index]


class SWENet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.g = torch.tensor(1.).float().to(device)
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.linear = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        for linear in self.linear:
            nn.init.xavier_normal_(linear.weight.data)

    def forward(self, a):
        for j in range(len(self.linear) - 1):
            z = self.linear[j](a)
            a = self.activation(z)
        z = self.linear[-1](a)
        return z

    def loss_ic(self, ic_batch):
        u_train, label = ic_batch
        u_hat = self.forward(u_train)
        return self.loss_func(u_hat, label)

    def loss_bc(self, bc_batch):
        u_train, v_label = bc_batch
        u_hat = self.forward(u_train)
        v_hat = u_hat[:, [1]]
        return self.loss_func(v_hat, v_label)

    def loss_physics(self, u_train):
        u_train.requires_grad = True
        nn_hat = self.forward(u_train)
        h_hat = nn_hat[:, [0]]
        v_hat = nn_hat[:, [1]]

        dh = autograd.grad(h_hat, u_train, torch.ones_like(h_hat), create_graph=True)[0]
        dv = autograd.grad(v_hat, u_train, torch.ones_like(v_hat), create_graph=True)[0]

        h_x = dh[:, [0]]
        h_t = dh[:, [1]]
        v_x = dv[:, [0]]
        v_t = dv[:, [1]]

        lhs_ = torch.hstack((h_t, h_hat * v_t + v_hat * h_t))
        rhs_ = torch.hstack(
            (-h_hat * v_x - v_hat * h_x, -2 * h_hat * v_hat * v_x - (v_hat ** 2 + self.g * h_hat) * h_x))

        return self.loss_func(lhs_, rhs_)

    def loss(self, ic_batch, bc_batch, physics_batch):
        loss_ic = self.loss_ic(ic_batch)
        loss_bc = self.loss_bc(bc_batch)
        loss_physics = self.loss_physics(physics_batch)
        return loss_ic + loss_bc + loss_physics


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    mu = np.linspace(0, 0.1, 11)[:, None]
    x = np.linspace(0, 1, 101)

    h0_train = 0.1 + 0.1 * np.exp(-64 * (x - mu) ** 2)
    v0_train = np.zeros_like(h0_train)
    ic_train = np.hstack((h0_train, v0_train))

    # 构建网络网络结构
    layers = [2 + ic_train.shape[1], 64, 64, 64, 64, 64, 64, 2]

    model = SWENet(layers)
    model.to(device)
    print("model:\n", model)

    x_train_ic = np.linspace(0, 1, 101)[:, None]
    t_train_ic = np.zeros_like(x_train_ic)
    y_train_ic = np.hstack((x_train_ic, t_train_ic))

    t_train_bc = np.linspace(0, 1, 101)[:, None]
    x_train_bc1 = np.zeros_like(t_train_bc)
    x_train_bc2 = np.ones_like(t_train_bc)
    y_train_bc1 = np.hstack((x_train_bc1, t_train_bc))
    y_train_bc2 = np.hstack((x_train_bc2, t_train_bc))
    y_train_bc = np.vstack((y_train_bc1, y_train_bc2))

    ic_ds = SWEICDataset(ic_train, y_train_ic)
    bc_ds = SWEBCDataset(ic_train, y_train_bc)
    physics_ds = SWEPhysicsDataset(ic_train)
    print("ic_ds count:", len(ic_ds), "bc_ds count:", len(bc_ds), "physics_ds count:", len(physics_ds))

    batch_size = 10000
    print("batch_size:", batch_size)

    ic_loader = DataLoader(ic_ds, batch_size=batch_size, shuffle=True)
    bc_loader = DataLoader(bc_ds, batch_size=batch_size, shuffle=True)
    physics_loader = DataLoader(physics_ds, batch_size=batch_size, shuffle=True)

    ic_iter = iter(ic_loader)
    bc_iter = iter(bc_loader)
    physics_iter = iter(physics_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=5e-7, mode='min', factor=0.5,
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
            ic_batch = next(ic_iter)
        except StopIteration:
            ic_iter = iter(ic_loader)
            ic_batch = next(ic_iter)

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

        loss = model.loss(ic_batch, bc_batch, physics_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if batch % 100 == 0:
            print(datetime.now(), 'batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
        if loss.item() < 1E-5:
            print(datetime.now(), 'batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'swe_1d_don_tbc_ic_03_e-5.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
