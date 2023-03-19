import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
import matplotlib.pyplot as plt
from pyDOE import lhs

"""
normal paradigm
"""


class LInfiniteLoss(nn.Module):
    def __init__(self):
        super(LInfiniteLoss, self).__init__()

    def forward(self, y, y_hat):
        max_loss = torch.max(torch.abs(y - y_hat))
        return max_loss


class FCN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.loss_func = LInfiniteLoss()
        # self.loss_func = nn.MSELoss(reduction='mean')
        self.rnn = nn.RNN(input_size=1, hidden_size=32, num_layers=2, batch_first=True)
        self.linear = nn.Linear(in_features=32, out_features=2)
        nn.init.xavier_normal_(self.linear.weight.data)
        nn.init.zeros_(self.linear.bias.data)

    def forward(self, x):
        # 正规化
        x = (x - x_lb) / (x_ub - x_lb)
        x, _ = self.rnn(x)
        x = self.linear(x)
        return x

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
        # return loss1 + loss2
        max_loss = loss1 if loss1 > loss2 else loss2
        return max_loss

    def loss(self, x_bc, y_bc, x_pde):
        loss_bc = self.loss_bc(x_bc, y_bc)
        loss_pde = self.loss_pde(x_pde)
        # return loss_bc + loss_pde
        max_loss = loss_bc if loss_bc > loss_pde else loss_pde
        return max_loss


if __name__ == '__main__':
    # float is float32
    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    PINN = FCN()
    print(PINN)

    x_lb = 0.0
    x_ub = 2 * np.pi
    total_points = 1200
    x_test = torch.linspace(x_lb, x_ub, total_points)

    x_train_bc = torch.tensor([[0.0]])
    y_train_bc = torch.tensor([[0.0, 1.0]])

    # Nf = total_points // 2
    # x_train_nf = torch.from_numpy(x_lb + (x_ub - x_lb) * lhs(1, Nf)).float()
    # x_train_nf = torch.vstack((x_train_bc, x_train_nf))

    x_train_nf = x_test.unsqueeze(1)

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=2000, verbose=True)
    epoch = 0
    while True:
        epoch += 1
        loss = PINN.loss(x_train_bc, y_train_bc, x_train_nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)
        if epoch % 100 == 0:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())

        if loss.item() < 0.01:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    nn_predict = PINN(x_test[:, None]).detach().numpy()
    x_predict = nn_predict[:, 0]
    y_predict = nn_predict[:, 1]

    fig, ax = plt.subplots()
    ax.plot(x_test, x_predict, color='r', label='f1')
    ax.plot(x_test, y_predict, color='g', label='f2')
    ax.set_xlabel('x', color='black')
    ax.set_ylabel('f(x)', color='black')
    ax.legend(loc='upper left')
    plt.savefig('./figure/simple_ode_rnn_01.png')
    plt.show()
