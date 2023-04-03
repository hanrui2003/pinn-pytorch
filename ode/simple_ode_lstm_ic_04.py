import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs


class FCN(nn.Module):
    def __init__(self, x_lb, x_ub, y_train_ic, input_size=1, hidden_size=20, num_layers=2, out_features=1):
        super().__init__()
        self.x_lb = x_lb
        self.x_ub = x_ub
        self.y_train_ic = y_train_ic
        self.loss_func = nn.MSELoss(reduction='mean')
        self.prev_linear = nn.Linear(in_features=input_size, out_features=hidden_size)
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.post_linear = nn.Linear(in_features=hidden_size, out_features=out_features)

    def forward(self, x):
        # 正规化
        x = (x - self.x_lb) / (self.x_ub - self.x_lb)
        x = self.prev_linear(x)
        x, _ = self.lstm(x)
        x = self.post_linear(x)
        return x

    def loss(self, x_pde):
        x_pde.requires_grad = True
        y_hat = self.forward(x_pde)
        u_t = autograd.grad(y_hat, x_pde, torch.ones_like(y_hat), create_graph=True)[0]
        loss_pde = self.loss_func(u_t, torch.cos(x_pde))
        loss_ic = self.loss_func(self.y_train_ic, y_hat[0])
        return loss_pde + loss_ic


if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)
    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # 区间总数
    total_interval = 600
    # 总点数
    total_points = total_interval + 1
    # 区间长度
    h = 0.005
    x_lb = torch.tensor(0.)
    x_ub = torch.tensor(total_interval * h)
    y_train_ic = torch.tensor([0.])

    PINN = FCN(x_lb, x_ub, y_train_ic)
    PINN.to(device)
    print(PINN)

    x_test = torch.linspace(x_lb, x_ub, total_points)

    # 配置点
    x_train_nf = x_test.unsqueeze(1)

    x_train_nf = x_train_nf.float().to(device)

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=40000,
                                                           verbose=True)
    epoch = 0
    while True:
        epoch += 1
        with torch.backends.cudnn.flags(enabled=False):
            loss = PINN.loss(x_train_nf)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 100 == 0:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())

        if loss.item() < 0.001:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(PINN, 'simple_ode_lstm_ic_04.pt')
    PINN.cpu()
    nn_predict = PINN(x_test[:, None]).detach().numpy()
    # x_nn = nn_predict[:, [0]]
    # y_nn = nn_predict[:, [1]]
    # z_nn = nn_predict[:, [2]]

    fig, ax = plt.subplots()
    ax.plot(x_test, nn_predict, color='r', label='x')
    # ax.plot(x_test, y_nn, color='g', label='y')
    # ax.plot(x_test, z_nn, color='b', label='z')
    ax.set_title('$sin x$')
    ax.set_xlabel('t', color='black')
    ax.set_ylabel('f(t)', color='black', rotation=0)
    ax.legend(loc='upper right')
    plt.savefig('./figure/simple_ode_lstm_ic_04.png')
    # plt.show()
