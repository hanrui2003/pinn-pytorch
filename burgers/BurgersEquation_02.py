import torch
import torch.autograd as autograd  # computation graph
from torch import Tensor  # tensor node in the computation graph
import torch.nn as nn  # neural networks
import torch.optim as optim  # optimizers e.g. gradient descent, ADAM, etc.

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.ticker

import numpy as np
import time
from pyDOE import lhs  # Latin Hypercube Sampling
import scipy.io


def plot3D(x, t, y):
    x_plot = x.squeeze(1)
    t_plot = t.squeeze(1)
    X, T = torch.meshgrid(x_plot, t_plot)
    F_xt = y
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(T, X, F_xt, 20, cmap="rainbow")
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.show()


def plot3D_Matrix(x, t, y):
    X, T = x, t
    F_xt = y
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(T, X, F_xt, 20, cmap="rainbow")
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()
    ax = plt.axes(projection='3d')
    ax.plot_surface(T.numpy(), X.numpy(), F_xt.numpy(), cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.show()


def solutionplot(u_pred, X_u_train, u_train):
    fig, ax = plt.subplots()
    ax.axis('off')

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(u_pred, interpolation='nearest', cmap='rainbow',
                  extent=[T.min(), T.max(), X.min(), X.max()],
                  origin='lower', aspect='auto')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(X_u_train[:, 1], X_u_train[:, 0], 'kx', label='Data (%d points)' % (u_train.shape[0]), markersize=4,
            clip_on=False)

    line = np.linspace(x.min(), x.max(), 2)[:, None]
    ax.plot(t[25] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[50] * np.ones((2, 1)), line, 'w-', linewidth=1)
    ax.plot(t[75] * np.ones((2, 1)), line, 'w-', linewidth=1)

    ax.set_xlabel('$t$')
    ax.set_ylabel('$x$')
    ax.legend(frameon=False, loc='best')
    ax.set_title('$u(x,t)$', fontsize=10)

    ''' 
    Slices of the solution at points t = 0.25, t = 0.50 and t = 0.75
    '''

    ####### Row 1: u(t,x) slices ##################
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(x, usol.T[25, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[25, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.set_title('$t = 0.25s$', fontsize=10)
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(x, usol.T[50, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[50, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.50s$', fontsize=10)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), ncol=5, frameon=False)

    ax = plt.subplot(gs1[0, 2])
    ax.plot(x, usol.T[75, :], 'b-', linewidth=2, label='Exact')
    ax.plot(x, u_pred.T[75, :], 'r--', linewidth=2, label='Prediction')
    ax.set_xlabel('$x$')
    ax.set_ylabel('$u(x,t)$')
    ax.axis('square')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title('$t = 0.75s$', fontsize=10)

    plt.savefig('Burgers.png', dpi=500)


class FCN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        self.iter = 0
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data, gain=1.0)
            nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        if not torch.is_tensor(x):
            x = torch.from_numpy(x)
        u_b = torch.from_numpy(ub).float().to(device)
        l_b = torch.from_numpy(lb).float().to(device)
        # preprocessing input
        x = (x - l_b) / (u_b - l_b)  # feature scaling
        # convert to float
        a = x.float()
        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)
        return self.linears[-1](a)

    def loss_BC(self, x_bc, y_bc):
        y_hat = self.forward(x_bc)
        return self.loss_function(y_hat, y_bc)

    def loss_PDE(self, x_pde):
        g = x_pde.clone()
        g.requires_grad = True
        u = self.forward(g)
        u_x_t = autograd.grad(u, g, torch.ones([x_pde.shape[0], 1]), retain_graph=True, create_graph=True)[0]
        u_xx_tt = autograd.grad(u_x_t, g, torch.ones(x_pde.shape), create_graph=True)[0]
        u_x = u_x_t[:, [0]]
        u_t = u_x_t[:, [1]]
        u_xx = u_xx_tt[:, [0]]
        f = u_t + (self.forward(g)) * (u_x) - (nu) * u_xx

        loss_f = self.loss_function(f, f_hat)

        return loss_f

    def loss(self, x, y, X_train_Nf):
        loss_u = self.loss_BC(x, y)
        loss_f = self.loss_PDE(X_train_Nf)
        return loss_u + loss_f

    def closure(self):

        optimizer.zero_grad()

        loss = self.loss(X_train_Nu, U_train_Nu, X_train_Nf)

        loss.backward()

        self.iter += 1

        if self.iter % 100 == 0:
            error_vec, _ = PINN.test()

            print(loss, error_vec)

        return loss

    'test neural network'

    def test(self):

        u_pred = self.forward(X_test)

        error_vec = torch.linalg.norm((u - u_pred), 2) / torch.linalg.norm(u,
                                                                           2)  # Relative L2 Norm of the error (Vector)

        u_pred = u_pred.cpu().detach().numpy()

        u_pred = np.reshape(u_pred, (256, 100), order='F')

        return error_vec, u_pred


if "__main__" == __name__:
    # Set default dtype to float32
    torch.set_default_dtype(torch.float)
    torch.manual_seed(1234)
    np.random.seed(1234)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    steps = 10000
    lr = 1e-1
    layers = [2, 32, 32, 1]
    # Nu: Number of training points
    N_u = 100
    # Nf: Number of collocation points (Evaluate PDE)
    N_f = 10_000
    # diffusion coeficient
    nu = 0.01 / np.pi

    data = scipy.io.loadmat('./Data/Burgers.mat')
    x = data['x']  # 256 points between -1 and 1 [256x1]
    t = data['t']  # 100 time points between 0 and 1 [100x1]
    usol = data['usol']  # solution of 256x100 grid points

    # 注意 numpy和torch的meshgrid输出方向不同
    X, T = np.meshgrid(x, t)  # makes 2 arrays X and T such that u(X[i],T[j])=usol[i][j] are a tuple
    plot3D(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(usol))  # f_real was defined previously(function)

    print(x.shape, t.shape, usol.shape)
    print(X.shape, T.shape)

    X_test = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

    # Domain bounds
    lb = X_test[0]  # [-1. 0.]
    ub = X_test[-1]  # [1.  0.99]
    u_true = usol.flatten('F')[:, None]  # Fortran style (Column Major)
    print(lb, ub)

    '''Boundary Conditions'''

    # Initial Condition -1 =< x =<1 and t = 0
    # 注意一下的命名 bottom_X 表示x=0时，t的取值，之所以都叫X，站在神经网络的输入一般称之为X
    left_X = np.hstack((X[0, :][:, None], T[0, :][:, None]))  # L1
    left_U = usol[:, 0][:, None]

    # Boundary Condition x = -1 and 0 =< t =<1
    bottom_X = np.hstack((X[:, 0][:, None], T[:, 0][:, None]))  # L2
    bottom_U = usol[-1, :][:, None]

    # Boundary Condition x = 1 and 0 =< t =<1
    top_X = np.hstack((X[:, -1][:, None], T[:, 0][:, None]))  # L3
    top_U = usol[0, :][:, None]

    X_train = np.vstack([left_X, bottom_X, top_X])
    U_train = np.vstack([left_U, bottom_U, top_U])

    # choose random N_u points for training
    idx = np.random.choice(X_train.shape[0], N_u, replace=False)

    X_train_Nu = X_train[idx, :]  # choose indices from  set 'idx' (x,t)
    U_train_Nu = U_train[idx, :]  # choose corresponding u

    '''Collocation Points'''

    # Latin Hypercube sampling for collocation points
    # N_f sets of tuples(x,t)
    X_train_Nf = lb + (ub - lb) * lhs(2, N_f)
    X_train_Nf = np.vstack((X_train_Nf, X_train_Nu))  # append training points to collocation points

    print("Original shapes for X and U:", X.shape, usol.shape)
    print("Boundary shapes for the edges:", left_X.shape, bottom_X.shape, top_X.shape)
    print("Available training data:", X_train.shape, U_train.shape)
    print("Final training data:", X_train_Nu.shape, U_train_Nu.shape)
    print("Total collocation points:", X_train_Nf.shape)

    'Convert to tensor and send to GPU'
    X_train_Nf = torch.from_numpy(X_train_Nf).float().to(device)
    X_train_Nu = torch.from_numpy(X_train_Nu).float().to(device)
    U_train_Nu = torch.from_numpy(U_train_Nu).float().to(device)
    X_test = torch.from_numpy(X_test).float().to(device)
    u = torch.from_numpy(u_true).float().to(device)
    f_hat = torch.zeros(X_train_Nf.shape[0], 1).to(device)

    PINN = FCN(layers)

    PINN.to(device)

    'Neural Network Summary'
    print(PINN)

    params = list(PINN.parameters())

    '''Optimization'''

    'L-BFGS Optimizer'
    optimizer = torch.optim.LBFGS(PINN.parameters(), lr,
                                  max_iter=steps,
                                  max_eval=None,
                                  tolerance_grad=1e-11,
                                  tolerance_change=1e-11,
                                  history_size=100,
                                  line_search_fn='strong_wolfe')

    start_time = time.time()

    optimizer.step(PINN.closure)

    elapsed = time.time() - start_time
    print('Training time: %.2f' % (elapsed))

    ''' Model Accuracy '''
    error_vec, u_pred = PINN.test()

    print('Test Error: %.5f' % (error_vec))

    solutionplot(u_pred, X_train_Nu.cpu().detach().numpy(), U_train_Nu)

    x1 = X_test[:, 0]
    t1 = X_test[:, 1]

    arr_x1 = x1.reshape(shape=X.shape).transpose(1, 0).detach().cpu()
    arr_T1 = t1.reshape(shape=X.shape).transpose(1, 0).detach().cpu()
    arr_y1 = u_pred
    arr_y_test = usol

    plot3D_Matrix(arr_x1, arr_T1, torch.from_numpy(arr_y1))
    plot3D(torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(usol))
