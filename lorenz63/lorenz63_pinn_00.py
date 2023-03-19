import torch
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs
import torch.nn as nn
import torch.autograd as autograd

# 伪随机，便于重复实验，便于调试
# np.random.seed(1234)
# torch.manual_seed(1234)
# t 的个数
Nt = 3000
# 步长
h = 0.01
width = Nt * h
# 生成固定间隔的t，这里主要用于后续画图
t = torch.arange(0, width, h)[:, None]
# lorenz 方程参数
rho = 15.0
sigma = 10.0
beta = 8.0 / 3.0

# 训练数据，在lorenz系统中，即初值条件
x_train_nu = torch.tensor([[0]])
y_train_nu = torch.tensor([[-8., 7., 27.]])
# y_train_nu = torch.tensor([[0.1, 0.0, 0.0]])

# 配置点个数
Nf = 1500
x_train_nf = torch.from_numpy(width * lhs(1, Nf)).float()
x_train_nf = torch.vstack((x_train_nf, x_train_nu))


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
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        a = x.float()
        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)
        return self.linears[-1](a)

    # 初值点损失
    def loss_bc(self, x_bc, y_bc):
        y_hat = self.forward(x_bc)
        return self.loss_function(y_bc, y_hat)

    # 配置点损失
    def loss_pde(self, x_pde):
        g = x_pde.clone()
        g.requires_grad = True
        # 网络输出
        y_hat = self.forward(g)
        x = y_hat[:, [0]]
        y = y_hat[:, [1]]
        z = y_hat[:, [2]]
        # 求一阶导
        x_t = autograd.grad(x, g, torch.ones_like(x), create_graph=True)[0]
        y_t = autograd.grad(y, g, torch.ones_like(y), create_graph=True)[0]
        z_t = autograd.grad(z, g, torch.ones_like(z), create_graph=True)[0]

        # f1 = sigma * (y - x)
        # f2 = x * (rho - z) - y
        # f3 = x * y - beta * z
        f1 = sigma * x
        f2 = y
        f3 = beta * z
        loss_f1 = self.loss_function(f1, x_t)
        loss_f2 = self.loss_function(f2, y_t)
        loss_f3 = self.loss_function(f3, z_t)
        return loss_f1 + loss_f2 + loss_f3

    def loss(self, x_bc, y_bc, x_pde):
        loss_bc = self.loss_bc(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        return loss_bc + loss_pde


layers = np.array([1, 32, 32, 32, 32, 3])
PINN = FCN(layers)
print(PINN)

optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-4, amsgrad=False)

epochs = 20000
for i in range(epochs):
    loss = PINN.loss(x_train_nu, y_train_nu, x_train_nf)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (i + 1) % 1000 == 0:
        print("epoch :", i, "loss :", loss.item())

nn_predict = PINN(t).detach().numpy()
x_predict = nn_predict[:, 0]
y_predict = nn_predict[:, 1]
z_predict = nn_predict[:, 2]

fig, ax = plt.subplots(1, 1)
ax.plot(t, x_predict, color='r', label='x')
ax.plot(t, y_predict, color='g', label='y')
ax.plot(t, z_predict, color='b', label='z')
ax.set_xlabel('t', color='black')
ax.set_ylabel('nn(x)', color='black')
ax.legend(loc='upper left')
plt.savefig('./figure/lorenz63_2d_00.png')
plt.show()

ax = plt.axes(projection='3d')
ax.plot(x_predict, y_predict, z_predict, 'r', label='Lorenz 63 model')
ax.legend()
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.savefig('./figure/lorenz63_3d_00.png')
plt.show()
