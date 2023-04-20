import numpy as np
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
import itertools


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
        # 分段线性差值
        func_list.append(lambda x0, y=y_sample: np.interp(x0, x.flatten(), y))

    return func_list


class ADRICBCDataset(Dataset):
    """
    初边值点对应的训练集
    """

    def __init__(self, ic_func_list, net_input_size, ic_train_count=100, each_bc_train_count=100):
        """
        根据随机函数，初始化数据集
        :param ic_func_list: 函数列表
        :param net_input_size: 网络输入维度，用于将随机函数离散化表示，确定离散的点数
        :param ic_train_count:随机生成的初值训练点
        :param each_bc_train_count:随机生成的初值训练点（上、下边界）
        """
        func_count = len(ic_func_list)
        icbc_count = ic_train_count + each_bc_train_count * 2
        self.length = func_count * icbc_count
        # 取一列x的离散值，然后获取对应的函数值，作为网络输入
        func_x = np.linspace(0, 1, net_input_size)
        func_y = [func(func_x) for func in ic_func_list]
        # 根据初值训练点的个数，构造训练数据
        u0_train_ic = np.repeat(func_y, ic_train_count, axis=0)

        # 均匀采样
        x_ic = np.random.rand(ic_train_count, 1)
        t_ic = np.zeros((ic_train_count, 1))
        label_ic = [func(x_ic) for func in ic_func_list]
        label_ic = np.asarray(label_ic).reshape(-1, 1)

        y_ic = np.hstack((x_ic, t_ic))
        y_train_ic = np.tile(y_ic, (func_count, 1))

        # # 根据边值训练点的个数，构造训练数据
        u0_train_bc = np.repeat(func_y, each_bc_train_count * 2, axis=0)

        # 均匀采样
        x_lbc = np.zeros((each_bc_train_count, 1))
        x_ubc = np.ones((each_bc_train_count, 1))
        t_bc = np.random.rand(each_bc_train_count * 2, 1)

        x_bc = np.vstack((x_lbc, x_ubc))
        y_bc = np.hstack((x_bc, t_bc))
        y_train_bc = np.tile(y_bc, (func_count, 1))
        label_bc = np.zeros((y_train_bc.shape[0], 1))

        u0_train = np.vstack((u0_train_ic, u0_train_bc))
        y_train = np.vstack((y_train_ic, y_train_bc))
        label = np.vstack((label_ic, label_bc))

        self.u0_train = torch.from_numpy(u0_train).float()
        self.y_train = torch.from_numpy(y_train).float()
        self.label = torch.from_numpy(label).float()
        # print("u0_train", self.u0_train)
        # print("y_train", self.y_train)
        # print("label", self.label)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.u0_train[index], self.y_train[index], self.label[index]


class ADRPhysicsDataset(Dataset):
    """
    物理信息对应的训练集
    """

    def __init__(self, ic_func_list, net_input_size, physics_train_count=100):
        """
        根据随机函数，初始化数据集
        :param ic_func_list: 函数列表
        :param net_input_size: 网络输入维度，用于将随机函数离散化表示，确定离散的点数
        :param physics_train_count:随机生成的训练点
        """
        func_count = len(ic_func_list)
        self.length = func_count * physics_train_count
        # 取一列x的离散值，然后获取对应的函数值，作为网络输入
        func_x = np.linspace(0, 1, net_input_size)
        func_y = [func(func_x) for func in ic_func_list]
        # 根据训练点的个数，构造训练数据
        u0_train_physics = np.repeat(func_y, physics_train_count, axis=0)

        # 均匀采样
        y_physics = np.random.rand(physics_train_count, 2)
        y_train_physics = np.tile(y_physics, (func_count, 1))

        self.u0_train = torch.from_numpy(u0_train_physics).float()
        self.y_train = torch.from_numpy(y_train_physics).float()
        # print("u0_train", self.u0_train)
        # print("y_train", self.y_train)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.u0_train[index], self.y_train[index]


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

    def loss_icbc(self, icbc_batch):
        u0_train, y_train, label = icbc_batch
        u_hat = self.forward(u0_train, y_train)
        return self.loss_func(u_hat, label)

    def loss_physics(self, physics_batch):
        u0_train, y_train = physics_batch
        y_train.requires_grad = True
        u_hat = self.forward(u0_train, y_train)

        u_y = autograd.grad(u_hat, y_train, torch.ones_like(u_hat), create_graph=True)[0]
        u_yy = autograd.grad(u_y, y_train, torch.ones_like(u_y), create_graph=True)[0]

        u_t = u_y[:, [1]]
        u_xx = u_yy[:, [0]]

        return self.loss_func(u_t, 0.01 * u_xx + 0.01 * u_hat ** 2)

    def loss(self, icbc_batch, physics_batch):
        loss_icbc = self.loss_icbc(icbc_batch)
        loss_physics = self.loss_physics(physics_batch)
        return loss_icbc + loss_physics


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    # 随机生成的函数个数
    stoch_func_num = 5000

    # 构建网络网络结构
    branch_layers = [100, 20, 20, 20]
    trunk_layers = [2, 20, 20, 20]

    model = ADRNet(branch_layers, trunk_layers)
    print("model:\n", model)

    # 分支网络输入维度
    branch_input_size = branch_layers[0]
    # 主干网络输入维度
    trunk_input_size = trunk_layers[0]
    # 初值训练点个数
    ic_train_num = 100
    # 边值训练点个数
    each_bc_train_num = 100
    # 物理信息训练点个数
    physics_train_num = 100

    func_list = gp_sample(num=stoch_func_num)
    icbc_ds = ADRICBCDataset(func_list, branch_input_size, ic_train_num, each_bc_train_num)
    physics_ds = ADRPhysicsDataset(func_list, branch_input_size, physics_train_num)

    print("icbc_ds count:", len(icbc_ds), "physics_ds count:", len(physics_ds))

    batch_size = 10000
    icbc_loader = DataLoader(icbc_ds, batch_size=batch_size, shuffle=True)
    physics_loader = DataLoader(physics_ds, batch_size=batch_size, shuffle=True)

    icbc_iter = iter(icbc_loader)
    physics_iter = iter(physics_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=40000,
                                                           verbose=True)

    batch = 0
    while True:
        batch += 1

        # 从dataloader迭代器获取元素，如果到头了，就重新生成迭代器
        # 注意这里不能用itertools.cycle去实现，会导致shuffle失效，因为本质是同一个iter
        try:
            icbc_batch = next(icbc_iter)
        except StopIteration:
            icbc_iter = iter(icbc_loader)
            icbc_batch = next(icbc_iter)

        try:
            physics_batch = next(physics_iter)
        except StopIteration:
            physics_iter = iter(physics_loader)
            physics_batch = next(physics_iter)

        loss = model.loss(icbc_batch, physics_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if batch % 100 == 0:
            print('batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
        if loss.item() < 0.01:
            print('batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'adr_don_ic_01.pt')
