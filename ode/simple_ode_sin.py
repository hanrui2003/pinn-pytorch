import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs


class FCN(nn.Module):
    def __init__(self, layers) -> None:
        super().__init__()
        self.activation = nn.Tanh()
        self.loss_func = nn.MSELoss(reduction='mean')
        self.linears = nn.ModuleList([nn.Linear(layers[i], layers[i + 1]) for i in range(len(layers) - 1)])
        for linear in self.linears:
            nn.init.xavier_normal_(linear.weight.data)
            nn.init.zeros_(linear.bias.data)

    def forward(self, x):
        a = x.float()
        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)
        return self.linears[-1](a)

    def loss_bc(self, x_bc, y_bc):
        y_hat = self.forward(x_bc)
        return self.loss_func(y_bc, y_hat)

    def loss_pde(self, x_pde):
        x_pde.requires_grad = True
        nn_out = self.forward(x_pde)
        nn_t = autograd.grad(nn_out, x_pde, torch.ones_like(nn_out), create_graph=True)[0]
        return self.loss_func(nn_t, torch.cos(x_pde))

    def loss(self, x_bc, y_bc, x_pde):
        loss_bc = self.loss_bc(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        return loss_bc + loss_pde


if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)

    # float is float32
    torch.set_default_dtype(torch.float)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    layers = [1, 32, 32, 1]
    PINN = FCN(layers)
    print(PINN)

    t_lb = 0.0
    t_ub = 16 * np.pi
    total_points = 5000
    Nf = total_points

    x_test = torch.linspace(t_lb, t_ub, total_points)

    t_bc = torch.tensor([[0.0]])
    t_bc_label = torch.tensor([[0.0]])

    t_pde = torch.from_numpy(t_lb + (t_ub - t_lb) * lhs(1, Nf)).float()
    t_pde = torch.vstack((t_bc, t_pde))

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)

    epochs = 50000
    for i in range(epochs):
        loss = PINN.loss(t_bc, t_bc_label, t_pde)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i + 1) % 1000 == 0:
            print('loss :', loss.item())

    nn_predict = PINN(x_test[:, None]).detach().numpy()

    fig, ax = plt.subplots()

    ax.plot(x_test, nn_predict, color='r', label='sin t')
    ax.set_xlabel('x', color='black')
    ax.set_ylabel('f(x)', color='black')
    ax.legend(loc='upper left')
    plt.savefig('./figure/ODEs_02_2d.png')
    plt.show()
