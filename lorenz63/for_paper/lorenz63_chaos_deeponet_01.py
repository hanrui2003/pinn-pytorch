import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import torch
import torch.nn as nn
import torch.autograd as autograd
from torch.utils.data import Dataset, DataLoader
from datetime import datetime


def generate_ic(num=5000):
    """
    随机生成初值
    根据RK方法跑出来的数据，当做随机向量的样本，通过核密度估计，得到其密度函数，
    然后进行随机采样，得到一列初值
    :param num: 随机生成的初值个数
    :return:
    """
    U = np.load('lorenz63_chaos.npy')
    # 定义带宽范围
    bandwidths = 10 ** np.linspace(-1, 1, 100)
    # 网格搜索最优带宽
    grid = GridSearchCV(KernelDensity(kernel='gaussian'), {'bandwidth': bandwidths}, cv=5)
    grid.fit(U)
    best_bandwidth = grid.best_params_['bandwidth']
    print("best bandwidth", best_bandwidth)
    # 拟合KDE模型
    kde = KernelDensity(kernel='gaussian', bandwidth=best_bandwidth).fit(U)
    return kde.sample(num)


class L63ICDataset(Dataset):
    def __init__(self, ic):
        """
        生成初值训练数据
        :param ic: 初值
        """
        self.length = len(ic)

        # 初值对应的训练点，即为0
        t_train = np.zeros((ic.shape[0], 1))

        # 拼接输入
        s_train = np.hstack((t_train, ic))

        # 需要注意的是，只有在 NumPy 数组的数据类型和 Tensor 的数据类型完全一致时，torch.from_numpy() 才能成功共享内存，
        # 否则它会在创建时进行数据类型的转换，此时就会开辟新的内存空间
        # 这里numpy默认float64，而torch默认float32，所以会开辟新空间
        self.s_train = torch.from_numpy(s_train).float().to(device)
        self.label = torch.from_numpy(ic).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.s_train[index], self.label[index]


class L63PhysicsDataset(Dataset):
    def __init__(self, ic, physics_train_count=21):
        """
        生成初值训练数据
        :param ic: 初值
        """
        ic_count = len(ic)
        self.length = ic_count * physics_train_count

        u0_train = np.repeat(ic, physics_train_count, axis=0)

        # 均匀取点[0,0.1]
        t_train = np.linspace(0, 0.1, physics_train_count)[:, None]
        t_train = np.tile(t_train, (ic_count, 1))

        # 拼接输入
        s_train = np.hstack((t_train, u0_train))

        self.s_train = torch.from_numpy(s_train).float().to(device)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        return self.s_train[index]


class L63Net(nn.Module):
    def __init__(self, layers):
        super().__init__()
        # lorenz 方程参数
        self.rho = torch.tensor(28.0)
        self.sigma = torch.tensor(10.0)
        self.beta = torch.tensor(8.0 / 3.0)
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def forward(self, a):
        for j in range(len(self.linears) - 1):
            z = self.linears[j](a)
            a = self.activation(z)
        z = self.linears[-1](a)
        return z

    def loss_ic(self, ic_batch):
        s_train, label = ic_batch
        u_hat = self.forward(s_train)
        return self.loss_func(u_hat, label)

    def loss_physics(self, s_train):
        s_train.requires_grad = True
        u_hat = self.forward(s_train)
        x_hat = u_hat[:, [0]]
        y_hat = u_hat[:, [1]]
        z_hat = u_hat[:, [2]]

        dx = autograd.grad(x_hat, s_train, torch.ones_like(x_hat), create_graph=True)[0]
        dy = autograd.grad(y_hat, s_train, torch.ones_like(y_hat), create_graph=True)[0]
        dz = autograd.grad(z_hat, s_train, torch.ones_like(z_hat), create_graph=True)[0]

        x_t = dx[:, [0]]
        y_t = dy[:, [0]]
        z_t = dz[:, [0]]

        lhs_ = torch.hstack((x_t, y_t, z_t))
        rhs_ = torch.hstack(
            (self.sigma * (y_hat - x_hat), x_hat * (self.rho - z_hat) - y_hat, x_hat * y_hat - self.beta * z_hat))

        return self.loss_func(lhs_, rhs_)

    def loss(self, ic_batch, physics_batch):
        loss_ic = self.loss_ic(ic_batch)
        loss_physics = self.loss_physics(physics_batch)
        return loss_ic + loss_physics


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("device", device)

    # 随机生成的初值训练点个数
    stoch_ic_count = 10000
    ic_raw = generate_ic(stoch_ic_count)
    ic_ds = L63ICDataset(ic_raw)

    # 物理信息训练点个数
    physics_ds = L63PhysicsDataset(ic_raw)

    print("ic_ds count:", len(ic_ds), "physics_ds count:", len(physics_ds))

    ic_loader = DataLoader(ic_ds, batch_size=stoch_ic_count // 10, shuffle=True)
    physics_loader = DataLoader(physics_ds, batch_size=10000, shuffle=True)

    ic_iter = iter(ic_loader)
    physics_iter = iter(physics_loader)

    # 构建网络网络结构
    layers = [4, 64, 64, 64, 64, 64, 3]

    model = L63Net(layers)
    model.to(device)
    print("model:\n", model)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=5000, verbose=True)
    # 记录训练开始时间
    start_time = datetime.now()
    print("Training started at:", start_time.strftime("%Y-%m-%d %H:%M:%S"))
    batch = 0
    save_threshold = 1e-3
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
        if loss.item() < save_threshold:
            print(datetime.now(), 'batch :', batch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(model, 'lorenz63_chaos_deeponet_01_' + str(save_threshold) + '.pt')

    # 训练结束后记录结束时间并计算总时间
    end_time = datetime.now()
    elapsed_time = end_time - start_time
    print("Training ended at:", end_time.strftime("%Y-%m-%d %H:%M:%S"))
    print("Elapsed time: ", elapsed_time)
