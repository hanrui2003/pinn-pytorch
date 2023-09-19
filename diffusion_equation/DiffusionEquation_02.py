import torch
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import torch.nn as nn
import torch.autograd as autograd


# def plot3D(x, t, y):
#     x_plot = x
#     t_plot = t
#     X, T = torch.meshgrid(x_plot, t_plot)
#     F_xt = y
#     fig, ax = plt.subplots(1, 1)
#     # 画20条等高线，彩虹渐变色
#     cp = ax.contourf(X, T, F_xt, 20, cmap="rainbow")
#     fig.colorbar(cp)
#     ax.set_title('F(x,t)')
#     ax.set_xlabel('t')
#     ax.set_ylabel('x')
#     plt.show()
#     ax = plt.axes(projection='3d')
#     ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
#     ax.set_xlabel('t')
#     ax.set_ylabel('x')
#     ax.set_zlabel('f(x,t)')
#     plt.show()


def plot3D_Matrix(x, t, y, path1, path2):
    X, T = x, t
    F_xt = y
    fig, ax = plt.subplots(1, 1)
    # 画20条等高线，彩虹渐变色
    cp = ax.contourf(T, X, F_xt, 20, cmap="rainbow")
    fig.colorbar(cp)
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.savefig(path1)
    # plt.close()
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.savefig(path2)
    # plt.close()
    plt.show()


def f_real(x, t):
    return torch.exp(-t) * (torch.sin(np.pi * x))


# TODO 调试时修改
x_grid_count = 200
t_grid_count = 100
Nu = 100
Nf = 10000
# 伪随机，便于重复实验，便于调试
# np.random.seed(1234)
# torch.manual_seed(1234)

# 取网格点，主要为了展示一下原来的函数，及后续的边界点选取
x_grid = torch.linspace(-1, 1, x_grid_count)
t_grid = torch.linspace(0, 1, t_grid_count)

# 计算函数值
X, T = torch.meshgrid(x_grid, t_grid)
y = f_real(X, T)
plot3D_Matrix(X, T, y, './figure/real_contour.png', './figure/real_surface.png')

# 准备神经网络数据
# x_test = torch.hstack((X.transpose(1, 0).flatten()[:, None], T.transpose(1, 0).flatten()[:, None]))
# y_test = y.transpose(1, 0).flatten()[:, None]
x_test = torch.hstack((X.flatten()[:, None], T.flatten()[:, None]))
y_test = y.flatten()[:, None]

# 坐标上下界 即 [-1., 0.] 和 [1., 1.] ， 用于后续采样的边界
lb = x_test[0]
ub = x_test[-1]

# 初值条件 t = 0, 即 T的第一列
left_x = torch.hstack((X[:, 0][:, None], T[:, 0][:, None]))
left_y = torch.sin(np.pi * left_x[:, 0]).unsqueeze(1)

# 边值条件 x=-1 和 x=1, 即X的第一行和最后一行
bottom_x = torch.hstack((X[-1, :][:, None], T[-1, :][:, None]))
bottom_y = torch.zeros(bottom_x.shape[0], 1)
# print(bottom_x)
# print(bottom_y)

top_x = torch.hstack((X[0, :][:, None], T[0, :][:, None]))
top_y = torch.zeros(top_x.shape[0], 1)
# print(top_x)
# print(top_y)

# 堆叠所有初边值数据
X_train = torch.vstack([left_x, bottom_x, top_x])
Y_train = torch.vstack([left_y, bottom_y, top_y])
# print(X_train)
# print(Y_train)


idx = np.random.choice(X_train.shape[0], Nu, replace=False)
X_train_Nu = X_train[idx, :]
Y_train_Nu = Y_train[idx, :]
# print(X_train_Nu, Y_train_Nu, sep='\n')

# 选取配置点(collocation points)数据用于网络训练
# 这里采用 拉丁超立方抽样
X_train_Nf = lb + (ub - lb) * lhs(2, Nf)
# print(X_train_Nf)
X_train_Nf = torch.vstack((X_train_Nf, X_train_Nu))
# print(X_train_Nf)

# 统一转换为float32，移至可用设备训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device :", device)
X_train_Nu = X_train_Nu.float().to(device)
Y_train_Nu = Y_train_Nu.float().to(device)
X_train_Nf = X_train_Nf.float().to(device)
f_hat = torch.zeros(X_train_Nf.shape[0], 1).to(device)


# 神经网络
class FCN(nn.Module):
    def __init__(self, layers):
        super(FCN, self).__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        # ModuleList的好处就是能像python list 使用索引，同时里面的module又能被注册到网络
        # 相比于 nn.Sequential ， ModuleList 需要实现forward，但是好处是灵活
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        # 参数初始化，主要是weight， Glorot normalization
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)
        return self.linears[-1](a)

    # 边界点损失
    def lossBC(self, x_BC, y_BC):
        loss_BC = self.loss_function(self.forward(x_BC), y_BC)
        return loss_BC

    # 配置点损失
    def lossPDE(self, x_PDE):
        # 克隆原始input，以免相互影响
        g = x_PDE.clone()
        g.requires_grad = True
        f = self.forward(g)
        # 求一阶和二阶导
        # f_x_t = autograd.grad(f, g, torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        # f_xx_tt = autograd.grad(f_x_t, g, torch.ones(g.shape).to(device), create_graph=True)[0]
        f_x_t = autograd.grad(f, g, torch.ones_like(f).to(device), retain_graph=True, create_graph=True)[0]
        f_xx_tt = autograd.grad(f_x_t, g, torch.ones_like(f_x_t).to(device), create_graph=True)[0]
        # print(f_x_t)
        # print(f_xx_tt)
        # 用list作 索引，不会降维
        f_t = f_x_t[:, [1]]
        f_xx = f_xx_tt[:, [0]]
        # [:, 1:] 索引方式，不会降维
        f = f_t - f_xx + torch.exp(-g[:, 1:]) * (
                torch.sin(np.pi * g[:, 0:1]) - np.pi ** 2 * torch.sin(np.pi * g[:, 0:1]))
        return self.loss_function(f, f_hat)

    def loss(self, x_BC, y_BC, x_PDE):
        loss_bc = self.lossBC(x_BC, y_BC)
        loss_pde = self.lossPDE(x_PDE)
        return loss_bc + loss_pde


# nn = FCN([2, 3, 4, 1])
# print(nn)
# print(nn(torch.tensor([1., 2.])))
# nn.lossPDE(X_train_Nu)

layers = np.array([2, 32, 64, 1])
PINN = FCN(layers)
PINN.to(device)
print(PINN)
print("PINN is on cuda :" + str(list(PINN.parameters())[0].is_cuda))
print("X_train_Nu is on cuda :" + str(X_train_Nu.is_cuda))

optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-2, amsgrad=False)

x_test = x_test.to(device)
y_test = y_test.to(device)

for i in range(20000):
    loss = PINN.loss(X_train_Nu, Y_train_Nu, X_train_Nf)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 2000 == 0:
        with torch.no_grad():
            test_loss = PINN.lossBC(x_test, y_test)
        print("train loss :", loss.item(), "/test loss :", test_loss.item())

# 可视化神经网络训练出来的函数
# 因为网络的输出值是向量，所以做个处理，转换为矩阵，绘图
y_predict = PINN(x_test)
x_feature = x_test[:, 0]
t_feature = x_test[:, 1]

x_matrix = X
t_matrix = T
y_matrix = y_predict.reshape(shape=[x_grid_count, t_grid_count]).detach().cpu()
print("error: ", np.linalg.norm(y_matrix - y) / np.linalg.norm(y))
# x_matrix = x_feature.reshape(shape=[11, 21]).detach().cpu()
# t_matrix = t_feature.reshape(shape=[11, 21]).detach().cpu()
# y_matrix = y_predict.reshape(shape=[11, 21]).detach().cpu()
# x_matrix = x_feature.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
# t_matrix = t_feature.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
# y_matrix = y_predict.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
plot3D_Matrix(x_matrix, t_matrix, y_matrix, './figure/predict_contour.png', './figure/predict_surface.png')
