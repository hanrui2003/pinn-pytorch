import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


def RBF(x1, x2, output_scale=1.0, length_scale=0.2):
    """
    高斯径向基函数，用于生成协方差矩阵
    :param x1:随机向量
    :param x2:随机向量
    """
    diffs = np.expand_dims(x1 / length_scale, 1) - np.expand_dims(x2 / length_scale, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


def gp_sample(num=5000, x_lb=0., x_ub=1.):
    """
    高斯过程采样，生成随机函数
    :param x_lb:x的下界
    :param x_ub:x的上界
    :param num:需要生成的函数个数
    :return:随机生成的函数列表
    """
    # 随机采样的向量维度，也是插值函数的点数
    N = 500
    # 扰动项，防止后续的 cholesky 分解过程中，有些元素非常小甚至为零时，导致平方根运算不稳定。
    jitter = 1e-10
    x = np.linspace(x_lb, x_ub, N)[:, None]
    # 经过核函数运算得到的协方差矩阵
    K = RBF(x, x)
    # cholesky 分解，得到下三角矩阵
    L = np.linalg.cholesky(K + jitter * np.eye(N))

    func_list = []
    for _ in range(num):
        # 生成N维标准正态分布的向量，与L内积，得到符合协方差为K的正态分布向量
        y_sample = np.dot(L, np.random.randn(N))
        # 把两端的值修改为0，这主要是因为边值为0，这就要求(0,0)和(1,0)的值为0，才能保持连续
        y_sample[[0, -1]] = 0.
        # 分段线性差值
        func_list.append(lambda x0, y=y_sample: np.interp(x0, x.flatten(), y))

    return func_list


def generate_obs(num=5000):
    """
    随机生成一些源项，然后得到数值解，
    对于每个数值解，在网格上选取一些点，把这些点对应的值作为一组观测数据。
    :param num:观测数据的数量
    :return:
    """
    func_list = gp_sample(num)

    # 以下是数值解
    Nx = 50
    Nt = 100
    D = 0.01
    k = 0.01

    x = np.linspace(0, 1, Nx)
    t = np.linspace(0, 1, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2
    print("grid ratio", D * dt / h2)

    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    # numpy的meshgrid参数，第一个是横轴，正方向向右，第二个是纵轴，正方向向下，坐标原点左上角；
    X, T = np.meshgrid(x, t)

    # 选取观测点的子网格，就是对大网格取子矩阵
    # x和t每隔多少个点取一个点
    x_interval = Nx // 10
    t_interval = Nt // 10
    idx_x = [i for i in range(x_interval, Nx, x_interval)]
    idx_t = [i for i in range(t_interval, Nt, t_interval)]

    x_train = X[idx_t][:, idx_x].flatten()[:, None]
    t_train = T[idx_t][:, idx_x].flatten()[:, None]
    y_train = np.hstack((x_train, t_train))

    # 观测训练数据
    o_train = []
    for f in func_list:
        U = np.zeros((Nt, Nx))
        for n in range(Nt - 1):
            u_n = U[n]
            u_next = u_n + dt * ((D / h2) * D2 @ u_n + k * u_n ** 2 + f(x))
            u_next[np.array([0, -1])] = 0.
            U[n + 1] = u_next

        o_train.append(U[idx_t][:, idx_x].flatten())

    return np.asarray(o_train), y_train, func_list


class ADRObsDataset(Dataset):
    """
    观测点对应的训练集
    """

    def __init__(self, o_train_raw, y_train_raw):
        """
        根据随机函数，初始化数据集
        :param o_train_raw: 观测值
        :param y_train_raw: 观测点
        """
        o_train_count = len(o_train_raw)
        y_train_count = len(y_train_raw)
        self.length = o_train_count * y_train_count

        o_train = np.repeat(o_train_raw, y_train_count, axis=0)
        y_train = np.tile(y_train_raw, (o_train_count, 1))
        label = o_train_raw.reshape(-1, 1)

        self.o_train = torch.from_numpy(o_train).float().to(device)
        self.y_train = torch.from_numpy(y_train).float().to(device)
        self.label = torch.from_numpy(label).float().to(device)
        # print("u0_train", self.u0_train)
        # print("y_train", self.y_train)
        # print("label", self.label)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.o_train[index], self.y_train[index], self.label[index]


class ADRPhysicsDataset(Dataset):
    """
    物理信息对应的训练集
    """

    def __init__(self, o_train_raw, func_list, physics_train_count=100):
        """
        根据随机函数，初始化数据集
        :param o_train_raw: 观测值
        :param physics_train_count:随机生成的训练点
        """
        o_train_count = len(o_train_raw)
        self.length = o_train_count * physics_train_count
        # 根据训练点的个数，构造训练数据
        o_train_physics = np.repeat(o_train_raw, physics_train_count, axis=0)

        # 均匀采样
        y_physics = np.random.rand(physics_train_count, 2)
        y_train_physics = np.tile(y_physics, (o_train_count, 1))

        x_physics = y_physics[:, [0]]
        f_train_physics = np.asarray([f(x_physics) for f in func_list]).reshape(-1, 1)

        self.o_train = torch.from_numpy(o_train_physics).float().to(device)
        self.y_train = torch.from_numpy(y_train_physics).float().to(device)
        self.f_train = torch.from_numpy(f_train_physics).float().to(device)
        # print("u0_train", self.u0_train)
        # print("y_train", self.y_train)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.o_train[index], self.y_train[index], self.f_train[index]


class ADRNet(nn.Module):
    def __init__(self, branch_layers, trunk_layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.branch_linear = nn.ModuleList(
            [nn.Linear(branch_layers[i], branch_layers[i + 1]) for i in range(len(branch_layers) - 1)])
        self.trunk_linear = nn.ModuleList(
            [nn.Linear(trunk_layers[i], trunk_layers[i + 1]) for i in range(len(trunk_layers) - 1)])

    def forward(self, a_branch, a_trunk):
        for j in range(len(self.branch_linear) - 1):
            z_branch = self.branch_linear[j](a_branch)
            a_branch = self.activation(z_branch)
        z_branch = self.branch_linear[-1](a_branch)

        for i in range(len(self.trunk_linear) - 1):
            z_trunk = self.trunk_linear[i](a_trunk)
            a_trunk = self.activation(z_trunk)
        z_trunk = self.trunk_linear[-1](a_trunk)

        # pytorch 的内积操作只适用于两个一维向量，而神经网络的训练是批量计算的；
        # 所以这里用一个变通的方式，批量计算内积
        dot = torch.sum(torch.mul(z_branch, z_trunk), dim=1, keepdim=True)
        return dot

    def loss_obs(self, obs_batch):
        o_train, y_train, label = obs_batch
        u_hat = self.forward(o_train, y_train)
        return self.loss_func(u_hat, label)

    def loss_physics(self, physics_batch):
        o_train, y_train, f_train = physics_batch
        y_train.requires_grad = True
        u_hat = self.forward(o_train, y_train)

        u_y = autograd.grad(u_hat, y_train, torch.ones_like(u_hat), create_graph=True)[0]
        u_yy = autograd.grad(u_y, y_train, torch.ones_like(u_y), create_graph=True)[0]

        u_t = u_y[:, [1]]
        u_xx = u_yy[:, [0]]

        return self.loss_func(u_t, 0.01 * u_xx + 0.01 * u_hat ** 2 + f_train)

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

    # 随机生成的函数个数
    stoch_obs_group_num = 5000
    o_train_raw, y_train_raw, func_list = generate_obs(stoch_obs_group_num)

    # 构建网络网络结构
    branch_layers = [o_train_raw.shape[1], 64, 64, 64, 64, 64]
    trunk_layers = [y_train_raw.shape[1], 64, 64, 64, 64, 64]

    model = ADRNet(branch_layers, trunk_layers)
    model.to(device)
    print("model:\n", model)

    obs_ds = ADRObsDataset(o_train_raw, y_train_raw)

    physics_train_num = 200
    physics_ds = ADRPhysicsDataset(o_train_raw, func_list, physics_train_num)

    print("obs_ds count:", len(obs_ds), "physics_ds count:", len(physics_ds))

    batch_size = 10000
    obs_loader = DataLoader(obs_ds, batch_size=batch_size, shuffle=True)
    physics_loader = DataLoader(physics_ds, batch_size=batch_size, shuffle=True)

    obs_iter = iter(obs_loader)
    physics_iter = iter(physics_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=10000,
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
        if loss.item() < 0.0001:
            print('batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'adr_don_obs_02_gzz_01.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
