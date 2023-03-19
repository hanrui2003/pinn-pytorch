import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

torch.manual_seed(1234)
np.random.seed(1234)

# float is float32
torch.set_default_dtype(torch.float)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

x_lb = 0.0
x_ub = 8 * np.pi
total_points = 500
x_test = torch.linspace(x_lb, x_ub, total_points)

x_train_bc = torch.tensor([[0.]])
y_train_bc = torch.tensor([[0., 1.]])

n_f = 250
x_train_pde = torch.from_numpy(x_lb + (x_ub - x_lb) * lhs(1, n_f)).float()

x_train_bc = x_train_bc.float().to(device)
y_train_bc = y_train_bc.float().to(device)
x_train_pde = x_train_pde.float().to(device)


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
        # 正规化
        a = (x - x_lb) / (x_ub - x_lb)
        for i in range(len(self.linears) - 1):
            z = self.linears[i](a)
            a = self.activation(z)
        return self.linears[-1](a)

    def loss_bc(self, t_bc, t_bc_label):
        y_hat = self.forward(t_bc)
        return self.loss_func(t_bc_label, y_hat)

    def loss_pde(self, x_pde):
        x_pde.requires_grad = True
        y_hat = self.forward(x_pde)
        x_nn = y_hat[:, [0]]
        y_nn = y_hat[:, [1]]
        x_t = autograd.grad(x_nn, x_pde, torch.ones_like(x_nn), create_graph=True)[0]
        y_t = autograd.grad(y_nn, x_pde, torch.ones_like(y_nn), create_graph=True)[0]
        loss_pde_x = self.loss_func(x_t, y_nn)
        loss_pde_y = self.loss_func(y_t, -x_nn)
        return loss_pde_x + loss_pde_y

    # def loss(self, x_bc, y_bc, x_pde):
    #     loss_bc = self.loss_bc(x_bc, y_bc)
    #     loss_pde = self.loss_pde(x_pde)
    #     return loss_bc + loss_pde


layers = [1, 32, 32, 2]
PINN = FCN(layers)
PINN.to(device)
print(PINN)


# optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)

def closure():
    optimizer.zero_grad()
    loss_bc = PINN.loss_bc(x_train_bc, y_train_bc)
    loss_bc_pde = PINN.loss_pde(x_train_bc)
    loss_other_pde = PINN.loss_pde(x_train_pde)
    loss = loss_bc + loss_bc_pde + loss_other_pde
    loss.backward()
    return loss


optimizer = torch.optim.LBFGS(PINN.parameters(), 1e-1,
                              max_iter=20,
                              max_eval=None,
                              tolerance_grad=1e-11,
                              tolerance_change=1e-11,
                              history_size=100,
                              line_search_fn='strong_wolfe')
optimizer.step(closure)

# epochs = 10000
# for i in range(epochs):
#     loss_bc = PINN.loss_bc(x_train_bc, y_train_bc)
#     loss_bc_pde = PINN.loss_pde(x_train_bc)
#     loss_other_pde = PINN.loss_pde(x_train_pde)
#     loss = loss_bc + loss_bc_pde + loss_other_pde
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     if (i + 1) % 1000 == 0:
#         print('epoch :', i + 1, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
#     if (i + 1) % 100000 == 0:
#         optimizer.param_groups[0]['lr'] *= 0.5

nn_predict = PINN(x_test[:, None]).detach().numpy()
x_nn = nn_predict[:, [0]]
y_nn = nn_predict[:, [1]]

fig, ax = plt.subplots()
ax.plot(x_test, x_nn, color='r', label='sin 4t')
ax.plot(x_test, y_nn, color='g', label='cos 4t')
ax.set_title('sin4t & cos4t & -cos8t dependent, x∈[0,4π]')
ax.set_xlabel('x', color='black')
ax.set_ylabel('f(x)', color='black', rotation=0)
ax.legend(loc='upper left')
plt.savefig('./figure/ode_limit_11.png')
plt.show()
