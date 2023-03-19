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
# from sklearn.model_selection import train_test_split

import numpy as np
import time
from pyDOE import lhs  # Latin Hypercube Sampling
import scipy.io

# Set default dtype to float32
torch.set_default_dtype(torch.float)

# PyTorch random number generator
torch.manual_seed(1234)
# Random number generators in other libraries
np.random.seed(1234)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)


def f_BC(x):
    return torch.sin(x)


def PDE(x):  # The PDE equation. We use it to get the residual in the Neurl Network.
    return torch.cos(x)


class FCN(nn.Module):
    ##Neural Network
    def __init__(self, layers):
        super().__init__()  # call __init__ from parent class
        self.activation = nn.Tanh()
        self.loss_function = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for i in range(len(layers) - 1):
            nn.init.xavier_normal_(self.linears[i].weight.data, gain=1.0)
            nn.init.zeros_(self.linears[i].bias.data)

    def forward(self, x):
        if torch.is_tensor(x) != True:
            x = torch.from_numpy(x)
        a = x.float()
        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)
        a = self.linears[-1](a)
        return a

    # Loss BC
    def lossBC(self, x_BC):
        y_z_hat = self.forward(x_BC)
        y_real = torch.sin(x_BC)
        z_real = torch.cos(x_BC)
        y_z_real = torch.hstack((y_real, z_real))
        loss_BC = self.loss_function(y_z_hat, y_z_real)
        return loss_BC

    # Loss PDE
    def lossPDE(self, x_PDE):
        g = x_PDE.clone()
        g.requires_grad = True  # Enable differentiation
        f = self.forward(g)
        y_hat = f[:, 0][:, None]
        z_hat = f[:, 1][:, None]
        y_x = autograd.grad(y_hat, g, torch.ones_like(y_hat).to(device), retain_graph=True, create_graph=True)[0]
        z_x = autograd.grad(z_hat, g, torch.ones_like(z_hat).to(device), retain_graph=True, create_graph=True)[0]
        loss_PDE_1 = self.loss_function(y_x, PDE(g))
        loss_PDE_2 = self.loss_function(z_x, -torch.sin(g))
        return loss_PDE_1 + loss_PDE_2

    def loss(self, x_BC, x_PDE):
        loss_bc = self.lossBC(x_BC)
        loss_pde = self.lossPDE(x_PDE)
        return loss_bc + loss_pde


steps = 5000
lr = 1e-3
layers = np.array([1, 32, 64, 2])  # 5 hidden layers
min = 0
max = 2 * np.pi
total_points = 500
# Nu: Number of training points (2 as we only have 2 boundaries),
# Nf: Number of collocation points (Evaluate PDE)
Nu = 2
Nf = 250

# 画真解图
x = torch.linspace(min, max, total_points).view(-1, 1)  # prepare to NN
y_real = f_BC(x)
fig, ax1 = plt.subplots()
ax1.plot(x.detach().numpy(), y_real.detach().numpy(), color='blue', label='Real_Train')
# ax1.plot(x_train.detach().numpy(),yh.detach().numpy(),color='red',label='Pred_Train')
ax1.set_xlabel('x', color='black')
ax1.set_ylabel('f(x)', color='black')
ax1.tick_params(axis='y', color='black')
ax1.legend(loc='upper left')
# plt.show()

BC_1 = x[0, :]
BC_2 = x[-1, :]
# Total Training points BC1+BC2
all_train = torch.vstack([BC_1, BC_2])
# Select Nu points
idx = np.random.choice(all_train.shape[0], Nu, replace=False)
x_BC = all_train[idx]
print("x_BC :", x_BC)
# Select Nf points
# Latin Hypercube sampling for collocation points
x_PDE = BC_1 + (BC_2 - BC_1) * lhs(1, Nf)
# x_PDE = torch.vstack((x_PDE, x_BC))

# Store tensors to GPU
x_PDE = x_PDE.float().to(device)
x_BC = x_BC.to(device)
# Create Model
model = FCN(layers)
print(model)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, amsgrad=False)
start_time = time.time()

for i in range(steps):
    # y_hat = model(x_PDE)
    loss = model.loss(x_BC, x_PDE)  # use mean squared error
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (i + 1) % (steps / 10) == 0:
        print("loss :", loss.item())

# Function
y_hat = model(x.to(device))[:, 0][0:None]
y_real = f_BC(x)
z_hat = model(x.to(device))[:, 1][0:None]
# Error
print(model.lossBC(x.to(device)))

# Derivative
# g = x.to(device)
# g = g.clone()
# g.requires_grad = True  # Enable differentiation
# f = model(g)
# f_x = autograd.grad(f, g, torch.ones([g.shape[0], 1]).to(device), retain_graph=True, create_graph=True)[0]

y_real_plot = y_real.detach().numpy()
y_hat_plot = y_hat.detach().cpu().numpy() + 0.05
z_hat_plot = z_hat.detach().cpu().numpy()
# f_x_plot = f_x.detach().cpu().numpy()

# Plot
fig, ax1 = plt.subplots()
ax1.plot(x, y_real_plot, color='blue', label='Real')
ax1.plot(x, y_hat_plot, color='red', label='Predicted1')
ax1.plot(x, z_hat_plot, color='g', label='Predicted2')
# ax1.plot(x, f_x_plot, color='green', label='Derivative')
ax1.set_xlabel('x', color='black')
ax1.set_ylabel('f(x)', color='black')
ax1.tick_params(axis='y', color='black')
ax1.legend(loc='upper left')
plt.show()
