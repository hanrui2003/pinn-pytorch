import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
import sys
from pyDOE import lhs
import os


class SWEICDataset(Dataset):
    """
    初值训练集
    """

    def __init__(self, branch_train):
        x_train_ic = np.linspace(0, 1, 51)[:, None]
        t_train_ic = np.zeros_like(x_train_ic)
        y_train = np.hstack((x_train_ic, t_train_ic))

        branch_train_count = len(branch_train)
        y_train_count = len(y_train)
        self.length = branch_train_count * y_train_count

        # 最终神经网络的输入
        branch_train_nn = np.repeat(branch_train, y_train_count, axis=0)
        y_train_nn = np.tile(y_train, (branch_train_count, 1))

        # 拼接输入
        z_train_nn = np.hstack((y_train_nn, branch_train_nn))

        h_label = branch_train[:, :51].reshape(-1, 1)
        v_label = np.zeros_like(h_label)
        label = np.hstack((h_label, v_label))
        print('obs dataset size (mb):', (sys.getsizeof(z_train_nn) + sys.getsizeof(label)) / 1024 / 1024)

        self.z_train_nn = torch.from_numpy(z_train_nn).float().to(device)
        self.label = torch.from_numpy(label).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.z_train_nn[index], self.label[index]


class SWEObsDataset(Dataset):
    """
    初值训练集
    """

    def __init__(self, branch_train):
        t_train_obs = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, 1.])[:, None]
        x_train_obs_1 = 0.2 * np.ones_like(t_train_obs)
        x_train_obs_2 = 0.4 * np.ones_like(t_train_obs)
        x_train_obs_3 = 0.6 * np.ones_like(t_train_obs)
        x_train_obs_4 = 0.8 * np.ones_like(t_train_obs)
        y_train_obs_1 = np.hstack((x_train_obs_1, t_train_obs))
        y_train_obs_2 = np.hstack((x_train_obs_2, t_train_obs))
        y_train_obs_3 = np.hstack((x_train_obs_3, t_train_obs))
        y_train_obs_4 = np.hstack((x_train_obs_4, t_train_obs))

        y_train = np.vstack((y_train_obs_1, y_train_obs_2, y_train_obs_3, y_train_obs_4))

        branch_train_count = len(branch_train)
        y_train_count = len(y_train)
        self.length = branch_train_count * y_train_count

        # 最终神经网络的输入
        branch_train_nn = np.repeat(branch_train, y_train_count, axis=0)
        y_train_nn = np.tile(y_train, (branch_train_count, 1))

        # 拼接输入
        z_train_nn = np.hstack((y_train_nn, branch_train_nn))

        h_label = branch_train[:, 51:91].reshape(-1, 1)
        v_label = branch_train[:, 91:].reshape(-1, 1)
        label = np.hstack((h_label, v_label))
        print('obs dataset size (mb):', (sys.getsizeof(z_train_nn) + sys.getsizeof(label)) / 1024 / 1024)

        self.z_train_nn = torch.from_numpy(z_train_nn).float().to(device)
        self.label = torch.from_numpy(label).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.z_train_nn[index], self.label[index]


class SWEBCDataset(Dataset):
    """
    初值训练集
    """

    def __init__(self, branch_train):
        t_train_bc = np.linspace(0, 1, 101)[:, None]
        x_train_bc1 = np.zeros_like(t_train_bc)
        x_train_bc2 = np.ones_like(t_train_bc)
        y_train_bc1 = np.hstack((x_train_bc1, t_train_bc))
        y_train_bc2 = np.hstack((x_train_bc2, t_train_bc))
        y_train_bc = np.vstack((y_train_bc1, y_train_bc2))

        branch_train_count = len(branch_train)
        y_train_count = len(y_train_bc)
        self.length = branch_train_count * y_train_count

        branch_train_nn = np.repeat(branch_train, y_train_count, axis=0)
        y_train_nn = np.tile(y_train_bc, (branch_train_count, 1))

        # 拼接输入
        z_train_nn = np.hstack((y_train_nn, branch_train_nn))

        v_label = np.zeros((y_train_nn.shape[0], 1))

        print('bc dataset size (mb):', (sys.getsizeof(z_train_nn) + sys.getsizeof(v_label)) / 1024 / 1024)

        self.z_train_nn = torch.from_numpy(z_train_nn).float().to(device)
        self.v_label = torch.from_numpy(v_label).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.z_train_nn[index], self.v_label[index]


class SWEPhysicsDataset(Dataset):
    """
    物理信息训练集
    """

    def __init__(self, branch_train, physics_train_count=2000):
        branch_train_count = len(branch_train)
        self.length = branch_train_count * physics_train_count

        branch_train_nn = np.repeat(branch_train, physics_train_count, axis=0)

        y_physics = lhs(2, physics_train_count)
        y_train_nn = np.tile(y_physics, (branch_train_count, 1))

        # 拼接输入
        z_train_nn = np.hstack((y_train_nn, branch_train_nn))

        print('physics dataset size (mb):', sys.getsizeof(z_train_nn) / 1024 / 1024)

        self.z_train_nn = torch.from_numpy(z_train_nn).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.z_train_nn[index]


class SWENet(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])

        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)

    def forward(self, a):
        for j in range(len(self.linears) - 1):
            z = self.linears[j](a)
            a = self.activation(z)
        z = self.linears[-1](a)
        return z

    def loss_ic(self, ic_batch):
        z_train, label = ic_batch
        u_hat = self.forward(z_train)
        return self.loss_func(u_hat, label)

    def loss_obs(self, obs_batch):
        z_train, label = obs_batch
        u_hat = self.forward(z_train)
        return self.loss_func(u_hat, label)

    def loss_bc(self, bc_batch):
        z_train, v_label = bc_batch
        u_hat = self.forward(z_train)
        v_hat = u_hat[:, [1]]
        return self.loss_func(v_hat, v_label)

    def loss_physics(self, z_train):
        z_train.requires_grad = True
        nn_hat = self.forward(z_train)
        h_hat = nn_hat[:, [0]]
        v_hat = nn_hat[:, [1]]

        dh = autograd.grad(h_hat, z_train, torch.ones_like(h_hat), create_graph=True)[0]
        dv = autograd.grad(v_hat, z_train, torch.ones_like(v_hat), create_graph=True)[0]

        h_x = dh[:, [0]]
        h_t = dh[:, [1]]
        v_x = dv[:, [0]]
        v_t = dv[:, [1]]

        lhs_ = torch.hstack((h_t, h_hat * v_t + v_hat * h_t))
        rhs_ = torch.hstack(
            (-h_hat * v_x - v_hat * h_x, -2 * h_hat * v_hat * v_x - (v_hat ** 2 + h_hat) * h_x))

        return self.loss_func(lhs_, rhs_)

    # def loss(self, ic_batch, obs_batch, bc_batch, physics_batch):
    #     loss_ic = self.loss_ic(ic_batch)
    #     loss_obs = self.loss_obs(obs_batch)
    #     loss_bc = self.loss_bc(bc_batch)
    #     loss_physics = self.loss_physics(physics_batch)
    #     return 0.1 * loss_ic + loss_obs + loss_bc + loss_physics


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 前900个作为
    total_data = np.load('swe_1d_rbf_sample_4000_l_0.2_t_1.npy')
    train_data = total_data[:900]
    noise_data = total_data[1000:1900]

    # 观测位置选择x=0.5，对应的高度和速度的索引为
    # 遍历每个样本序列
    obs_index = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    ic_index = [i * 2 for i in range(51)]
    branch_data = np.empty((0, 131))
    for idx, seq in enumerate(train_data):
        # 噪声波的值域是[-0.01,0.01]，所以绝对误差在[0.,0.01]
        ic = seq[0] + 0.2 * noise_data[idx][0] - 0.03
        obs_h_1 = seq[obs_index, 20]
        obs_v_1 = seq[obs_index, 121]
        obs_h_2 = seq[obs_index, 40]
        obs_v_2 = seq[obs_index, 141]
        obs_h_3 = seq[obs_index, 60]
        obs_v_3 = seq[obs_index, 161]
        obs_h_4 = seq[obs_index, 80]
        obs_v_4 = seq[obs_index, 181]
        obs = np.concatenate((ic[ic_index], obs_h_1, obs_h_2, obs_h_3, obs_h_4, obs_v_1, obs_v_2, obs_v_3, obs_v_4))
        branch_data = np.vstack((branch_data, obs))

    # 构建网络网络结构
    layers = [133, 128, 128, 128, 128, 2]

    model = SWENet(layers)
    model.to(device)
    print("model:\n", model)

    old_model_weights_path = 'swe_1d_don_tbc_da_12_1e-3.pt'
    new_model_weights_path = 'swe_1d_don_tbc_da_12_2e-5.pt'
    save_threshold = 2e-5

    if os.path.exists(old_model_weights_path):
        # 如果权重文件存在，则加载权重
        model.load_state_dict(torch.load(old_model_weights_path))
        print(datetime.now(), "Loaded model weights from", old_model_weights_path)

    ic_ds = SWEICDataset(branch_data)
    obs_ds = SWEObsDataset(branch_data)
    bc_ds = SWEBCDataset(branch_data)
    physics_ds = SWEPhysicsDataset(branch_data)
    print("ic_ds count:", len(ic_ds), "obs_ds count:", len(obs_ds), "bc_ds count:", len(bc_ds), "physics_ds count:",
          len(physics_ds))

    batch_size = 10000
    ic_loader = DataLoader(ic_ds, batch_size=batch_size, shuffle=True)
    obs_loader = DataLoader(obs_ds, batch_size=batch_size, shuffle=True)
    bc_loader = DataLoader(bc_ds, batch_size=batch_size, shuffle=True)
    physics_loader = DataLoader(physics_ds, batch_size=batch_size, shuffle=True)

    ic_iter = iter(ic_loader)
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
            ic_batch = next(ic_iter)
        except StopIteration:
            ic_iter = iter(ic_loader)
            ic_batch = next(ic_iter)

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

        loss_ic = model.loss_ic(ic_batch)
        loss_obs = model.loss_obs(obs_batch)
        loss_bc = model.loss_bc(bc_batch)
        loss_physics = model.loss_physics(physics_batch)
        loss = 0.01 * loss_ic + loss_obs + loss_bc + loss_physics

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if batch % 100 == 0:
            print(datetime.now(), 'batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item(),
                  'loss_ic :', loss_ic.item(), 'loss_obs :', loss_obs.item(), 'loss_bc :', loss_bc.item(),
                  'loss_physics :', loss_physics.item())
        if loss.item() < save_threshold:
            print(datetime.now(), 'batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item(),
                  'loss_ic :', loss_ic.item(), 'loss_obs :', loss_obs.item(), 'loss_bc :', loss_bc.item(),
                  'loss_physics :', loss_physics.item())
            break

    torch.save(model.state_dict(), new_model_weights_path)

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
