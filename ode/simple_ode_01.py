import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

"""
normal paradigm
"""


class FCN(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[k], layers[k + 1]) for k in range(len(layers) - 1)])
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        # 正规化
        a = (x - x_lb) / (x_ub - x_lb)
        for j in range(len(self.linears) - 1):
            z = self.linears[j](a)
            a = self.activation(z)
        return self.linears[-1](a)

    def loss_bc(self, x_bc, y_bc):
        y_hat = self.forward(x_bc)
        return self.loss_func(y_bc, y_hat)

    def loss_pde(self, x_pde):
        x_pde.requires_grad = True
        y_hat = self.forward(x_pde)
        x_nn = y_hat[:, [0]]
        y_nn = y_hat[:, [1]]
        x_t = autograd.grad(x_nn, x_pde, torch.ones_like(x_nn), create_graph=True)[0]
        y_t = autograd.grad(y_nn, x_pde, torch.ones_like(y_nn), create_graph=True)[0]
        loss1 = self.loss_func(x_t, torch.cos(x_pde))
        loss2 = self.loss_func(y_t, -torch.sin(x_pde))
        return loss1 + loss2

    def loss(self, x_bc, y_bc, x_pde):
        loss_bc = self.loss_bc(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        return loss_bc + loss_pde


if __name__ == '__main__':
    # float is float32
    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    layers = [1, 32, 32, 2]
    PINN = FCN(layers)
    print(PINN)

    x_lb = 0.0
    x_ub = 2 * np.pi
    total_points = 500
    x_test = torch.linspace(x_lb, x_ub, total_points)

    x_train_bc = torch.tensor([[0.0]])
    y_train_bc = torch.tensor([[0.0, 1.0]])

    Nf = total_points // 2
    x_train_nf = torch.from_numpy(x_lb + (x_ub - x_lb) * lhs(1, Nf)).float()
    x_train_nf = torch.vstack((x_train_bc, x_train_nf))

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)

    epochs = 5000
    for i in range(epochs):
        loss = PINN.loss(x_train_bc, y_train_bc, x_train_nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 1000 == 0:
            print('loss :', loss.item())

    nn_predict = PINN(x_test[:, None]).detach().numpy()
    x_predict = nn_predict[:, 0]
    y_predict = nn_predict[:, 1]

    fig, ax = plt.subplots()
    ax.plot(x_test, x_predict, color='r', label='f1')
    ax.plot(x_test, y_predict, color='g', label='f2')
    ax.set_xlabel('x', color='black')
    ax.set_ylabel('f(x)', color='black')
    ax.legend(loc='upper left')
    plt.show()
