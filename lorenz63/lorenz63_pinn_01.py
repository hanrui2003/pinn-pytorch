import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

# torch.manual_seed(1234)
# np.random.seed(1234)

# float is float32
torch.set_default_dtype(torch.float)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# lorenz 方程参数
rho = torch.tensor(15.0)
sigma = torch.tensor(10.0)
beta = torch.tensor(8.0 / 3.0)


class FCN(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[j], layers[j + 1]) for j in range(len(layers) - 1)])
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        a = x.float()
        for j in range(len(self.linears) - 1):
            z = self.linears[j](a)
            a = self.activation(z)
        return self.linears[-1](a)

    def loss_bc(self, x_bc, y_bc):
        y_hat = self.forward(x_bc)
        return self.loss_func(y_bc, y_hat)

    def loss_pde(self, x_pde):
        x_pde.requires_grad = True
        nn_out = self.forward(x_pde)
        x = nn_out[:, [0]]
        y = nn_out[:, [1]]
        z = nn_out[:, [2]]
        x_t = autograd.grad(x, x_pde, torch.ones_like(x), create_graph=True)[0]
        y_t = autograd.grad(y, x_pde, torch.ones_like(y), create_graph=True)[0]
        z_t = autograd.grad(z, x_pde, torch.ones_like(z), create_graph=True)[0]
        f1 = sigma * (y - x) + torch.cos(x_pde)
        f2 = x * (rho - z) - y - torch.sin(x_pde)
        f3 = x * y - beta * z + torch.sin(x_pde)
        loss_x = self.loss_func(x_t, f1)
        loss_y = self.loss_func(y_t, f2)
        loss_z = self.loss_func(z_t, f3)
        # loss1 = self.loss_func(x_t, torch.cos(x_pde))
        # loss2 = self.loss_func(y_t, -torch.sin(x_pde))
        # loss3 = self.loss_func(z_t, torch.sin(x_pde))
        return loss_x + loss_y + loss_z

    def loss(self, x_bc, y_bc, x_pde):
        loss_bc = self.loss_bc(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        return loss_bc + loss_pde


layers = [1, 32, 32, 3]
PINN = FCN(layers)
print(PINN)

lb = torch.tensor(0.)
ub = torch.tensor(2 * np.pi)
total_points = 1000
x_test = torch.linspace(lb, ub, total_points)

x_train_bc = torch.tensor([[0.0]])
# y_train_bc = torch.tensor([[-8., 7., 27.]])
# y_train_bc = torch.tensor([[0., 1., -1.]])
y_train_bc = torch.tensor([[-8., 8., 26.]])

Nf = 500
x_train_nf = (lb + (ub - lb) * lhs(1, Nf)).float()
# 不要堆叠，边界点单独算，单独权重
# x_train_nf = torch.vstack((x_train_bc, x_train_nf))

optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30000, gamma=0.1)

epochs = 30000
for i in range(epochs):
    loss1 = PINN.loss_bc(x_train_bc, y_train_bc)
    loss2 = PINN.loss_pde(x_train_bc)
    loss3 = PINN.loss_pde(x_train_nf)
    loss = loss1 + loss2 + loss3
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    # scheduler.step()
    if (i + 1) % 1000 == 0:
        print('epoch :', i, 'lr :', optimizer.state_dict()['param_groups'][0]['lr'], 'loss :', loss.item(), 'loss1 :',
              loss1.item(), 'loss2 :', loss2.item(), 'loss3 :',
              loss3.item())

nn_predict = PINN(x_test[:, None]).detach().numpy()
x_predict = nn_predict[:, 0] - np.sin(x_test)
y_predict = nn_predict[:, 1] - np.cos(x_test)
z_predict = nn_predict[:, 2] + np.cos(x_test)

fig, ax = plt.subplots()
ax.plot(x_test, x_predict, color='r', label='f1')
ax.plot(x_test, y_predict, color='g', label='f2')
ax.plot(x_test, z_predict, color='b', label='f3')
ax.set_xlabel('x', color='black')
ax.set_ylabel('f(x)', color='black')
ax.legend(loc='upper left')
plt.savefig('./figure/lorenz63_2d_01.png')
plt.show()

ax = plt.axes(projection='3d')
ax.plot(x_predict, y_predict, z_predict, 'r', label='Lorenz 63 model')
ax.legend()
plt.xlabel('x')
plt.ylabel('y')
ax.set_zlabel('z')
plt.savefig('./figure/lorenz63_3d_01.png')
plt.show()
