import torch
import numpy as np
from matplotlib import pyplot as plt
# import matplotlib.pyplot as plt
from pyDOE import lhs
import torch.nn as nn
import torch.autograd as autograd


def f_real(x, t):
    return torch.exp(-t) * torch.sin(np.pi * x)


# 辅助函数，画图
def plot3D(x, t, y):
    # x_plot = x.squeeze(1)
    # t_plot = t.squeeze(1)
    X, T = torch.meshgrid(x, t)
    F_xt = y
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(T, X, F_xt, 20, cmap="rainbow")
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.savefig("./real1.png")
    # plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.savefig("./real2.png")
    # plt.show()


def plot3D_Matrix(x, t, y):
    X, T = x, t
    F_xt = y
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(T, X, F_xt, 20, cmap="rainbow")
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.savefig("./predict1.png")
    # plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.savefig("./predict2.png")
    # plt.show()


# print(f_real(torch.tensor(-0.5), torch.tensor(-1)))

# TODO
# x = torch.linspace(-1, 1, 20)
# t = torch.linspace(0, 1, 10)
x = torch.linspace(-1, 1, 200)
t = torch.linspace(0, 1, 100)
X, T = torch.meshgrid(x, t)
y_real = f_real(X, T)
plot3D(x, t, y_real)
print(X)
print(T)

x_test = torch.hstack((X.transpose(1, 0).flatten()[:, None], T.transpose(1, 0).flatten()[:, None]))
y_test = y_real.transpose(1, 0).flatten()[:, None]
# upper bound and lower bound
ub = x_test[-1]
lb = x_test[0]
# print(x_test.shape, y_test.shape)
# print(lb, ub)

# 初值条件
left_x = torch.hstack((X[:, 0][:, None], T[:, 0][:, None]))
left_y = torch.sin(np.pi * left_x[:, 0]).unsqueeze(1)
# print(left_x, left_y)

# 边界条件
bottom_x = torch.hstack((X[-1, :][:, None], T[-1, :][:, None]))
bottom_y = torch.zeros(bottom_x.shape[0], 1)
# print(bottom_x, bottom_y, sep='\n')

top_x = torch.hstack((X[0, :][:, None], T[0, :][:, None]))
top_y = torch.zeros(top_x.shape[0], 1)
# print(top_x, top_y, sep='\n')

X_train = torch.vstack([left_x, bottom_x, top_x])
Y_train = torch.vstack([left_y, bottom_y, top_y])

# TODO
# Nu = 10
Nu = 100
idx = np.random.choice(X_train.shape[0], Nu, replace=False)

X_train_Nu = X_train[idx, :]
Y_train_Nu = Y_train[idx, :]

# TODO
# Nf = 100
Nf = 10000
X_train_Nf = lb + (ub - lb) * lhs(2, Nf)
X_train_Nf = torch.vstack((X_train_Nf, X_train_Nu))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

X_train_Nu = X_train_Nu.float().to(device)
Y_train_Nu = Y_train_Nu.float().to(device)
X_train_Nf = X_train_Nf.float().to(device)
f_hat = torch.zeros(X_train_Nf.shape[0], 1).to(device)


# print(device)

class FCN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
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
        a = self.linears[-1](a)
        return a

    def lossBC(self, x_BC, y_BC):
        loss_BC = self.loss_function(self.forward(x_BC), y_BC)
        return loss_BC

    def lossPDE(self, x_PDE):
        g = x_PDE.clone()
        g.requires_grad = True
        f = self.forward(g)
        f_x_t = autograd.grad(f, g, torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]
        f_xx_tt = autograd.grad(f_x_t, g, torch.ones(g.shape).to(device), create_graph=True)[0]
        f_t = f_x_t[:, [1]]
        f_xx = f_xx_tt[:, [0]]
        f = f_t - f_xx + torch.exp(-g[:, 1:]) * torch.sin(np.pi * g[:, 0:1]) - np.pi ** 2 * torch.sin(np.pi * g[:, 0:1])
        return self.loss_function(f, f_hat)

    def loss(self, x_BC, y_BC, x_PDE):
        loss_bc = self.lossBC(x_BC, y_BC)
        loss_pde = self.lossPDE(x_PDE)
        return loss_bc + loss_pde


layers = np.array([2, 32, 64, 1])
PINN = FCN(layers)
PINN.to(device)
# print(PINN)
optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)

for i in range(20000):
    loss = PINN.loss(X_train_Nu, Y_train_Nu, X_train_Nf)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % (20000 / 10) == 0:
        with torch.no_grad():
            test_loss = PINN.lossBC(x_test, y_test)

    print("training:", loss, "Testing", test_loss)

y1 = PINN(x_test)
x1 = x_test[:, 0]
t1 = x_test[:, 1]

arr_x1 = x1.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
arr_T1 = t1.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
arr_y1 = y1.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
arr_y_test = y_test.reshape(shape=[100, 200]).transpose(1, 0).detach().cpu()
plot3D_Matrix(arr_x1, arr_T1, arr_y1)
