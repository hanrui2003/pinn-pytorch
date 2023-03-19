import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch
import torch.nn as nn
import torch.autograd as autograd


def plot3D(x, t, y):
    x = torch.from_numpy(x)
    t = torch.from_numpy(t)
    y = torch.from_numpy(y)
    x_plot = x.squeeze(1)
    t_plot = t.squeeze(1)
    X, T = torch.meshgrid(x_plot, t_plot)
    F_xt = y
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(T, X, F_xt, 20, cmap='rainbow')
    fig.colorbar(cp)
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.savefig('./figure/burgers_inverse_00_2d_real.png')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T, X, F_xt, cmap='rainbow')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.savefig('./figure/burgers_inverse_00_2d_predict.png')
    plt.show()


def plot3D_Matrix(x, t, y):
    X, T = x, t
    F_xt = y
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(T, X, F_xt, 20, cmap='rainbow')
    fig.colorbar(cp)
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T, X, F_xt, cmap='rainbow')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.show()


data = scipy.io.loadmat('./data/Burgers.mat')
x = data['x']
t = data['t']
usol = data['usol']

# plot3D(x, t, usol)

X, T = np.meshgrid(x, t)
X_true = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
U_true = usol.flatten('F')[:, None]

lb = X_true[0]
ub = X_true[-1]

total_points = len(x) * len(t)
N_u = 10000
idx = np.random.choice(total_points, N_u, replace=False)
X_train_Nu = X_true[idx]
U_train_Nu = U_true[idx]

# 统一转换为float32，移至可用设备训练
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device :", device)
X_train_Nu = torch.from_numpy(X_train_Nu).float().to(device)
U_train_Nu = torch.from_numpy(U_train_Nu).float().to(device)
X_true = torch.from_numpy(X_true).float().to(device)
U_true = torch.from_numpy(U_true).float().to(device)
u_b = torch.from_numpy(ub).float().to(device)
l_b = torch.from_numpy(lb).float().to(device)
f_hat = torch.zeros(X_train_Nu.shape[0], 1).to(device)


class DNN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        x = (x - l_b) / (u_b - l_b)
        a = x.float()
        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)
        return self.linears[-1](a)


lambda1 = 2.0
lambda2 = 0.02

nu = 0.01 / np.pi


class FCN():
    def __init__(self, layers):
        self.lambda1 = torch.tensor([lambda1], requires_grad=True).float().to(device)
        self.lambda2 = torch.tensor([lambda2], requires_grad=True).float().to(device)
        # 如果是在 Module 类里面，则自动注册为网络训练参数
        self.lambda1 = nn.Parameter(self.lambda1)
        self.lambda2 = nn.Parameter(self.lambda2)

        self.dnn = DNN(layers).to(device)
        # 这里是外部注册网络参数
        self.dnn.register_parameter('lambda1', self.lambda1)
        self.dnn.register_parameter('lambda2', self.lambda2)

        self.loss_function = nn.MSELoss(reduction='mean')
        self.iter = 0

    def loss_data(self, x, u):
        loss_u = self.loss_function(self.dnn(x), u)
        return loss_u

    def loss_PDE(self, X_train_Nu):
        lambda1 = self.lambda1
        lambda2 = self.lambda2
        g = X_train_Nu.clone()
        g.requires_grad = True
        u = self.dnn(g)
        u_x_t = autograd.grad(u, g, torch.ones_like(u).to(device), create_graph=True)[0]
        u_xx_tt = autograd.grad(u_x_t, g, torch.ones_like(u_x_t).to(device), create_graph=True)[0]
        u_x = u_x_t[:, [0]]
        u_t = u_x_t[:, [1]]
        u_xx = u_xx_tt[:, [0]]
        f = u_t + lambda1 * self.dnn(g) * u_x - lambda2 * u_xx
        loss_f = self.loss_function(f, f_hat)
        return loss_f

    def loss(self, x, y):
        loss_u = self.loss_data(x, y)
        loss_f = self.loss_PDE(x)
        loss = loss_u + loss_f
        return loss

    def closure(self):
        optimizer.zero_grad()
        loss = self.loss(X_train_Nu, U_train_Nu)
        loss.backward()
        self.iter += 1
        if self.iter % 100 == 0:
            error_vec, _ = PINN.test()
            print(
                'Relative Error(Test): %.5f , λ_real = [1.0, %.5f], λ_PINN = [%.5f, %.5f]',
                {
                    # error_vec.cpu().detach().numpy(),
                    '100/π',
                    self.lambda1.item(),
                    self.lambda2.item()
                }
            )
        return loss

    def test(self):
        u_pred = self.dnn(X_true)
        # Relative L2 Norm of the error (Vector)
        error_vec = torch.linalg.norm((U_true - u_pred), 2) / torch.linalg.norm(U_true, 2)
        u_pred = u_pred.cpu().detach().numpy()
        u_pred = np.reshape(u_pred, (x.shape[0], t.shape[0]), order='F')
        return error_vec, u_pred


layers = np.array([2, 20, 20, 20, 20, 20, 20, 20, 20, 1])
PINN = FCN(layers)
param = list(PINN.dnn.parameters())
optimizer = torch.optim.LBFGS(param, 1e-1,
                              max_iter=20000,
                              max_eval=None,
                              tolerance_grad=1e-11,
                              tolerance_change=1e-11,
                              history_size=100,
                              line_search_fn='strong_wolfe')
optimizer.step(PINN.closure)
