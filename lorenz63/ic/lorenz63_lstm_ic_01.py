import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np


class FCN(nn.Module):
    def __init__(self, pre_input_size=1, pre_hidden_size=32, pre_num_layers=2,
                 post_input_size=32, post_hidden_size=3, post_num_layers=1):
        super().__init__()

        self.y_train_ic = torch.tensor([-4., 7., 15]).to(device)
        self.loss_func = nn.MSELoss(reduction='mean')

        self.pre_lstm = nn.LSTM(input_size=pre_input_size, hidden_size=pre_hidden_size, num_layers=pre_num_layers,
                                batch_first=True)
        self.pre_h_0 = nn.Parameter(torch.randn(pre_num_layers, pre_hidden_size))
        self.pre_c_0 = nn.Parameter(torch.randn(pre_num_layers, pre_hidden_size))

        self.post_lstm = nn.LSTM(input_size=post_input_size, hidden_size=post_hidden_size, num_layers=post_num_layers,
                                 batch_first=True)
        self.post_h_0 = nn.Parameter(torch.randn(post_num_layers, post_hidden_size))
        self.post_c_0 = nn.Parameter(torch.randn(post_num_layers, post_hidden_size))

    def forward(self, x):
        # 正规化
        x = (x - x_lb) / (x_ub - x_lb)
        x, _ = self.pre_lstm(x, (self.pre_h_0, self.pre_c_0))
        x, _ = self.post_lstm(x, (self.post_h_0, self.post_c_0))
        return x

    def loss(self, x_pde):
        x_pde.requires_grad = True
        y_hat = self.forward(x_pde)
        x_nn = y_hat[:, [0]]
        y_nn = y_hat[:, [1]]
        z_nn = y_hat[:, [2]]
        x_t = autograd.grad(x_nn, x_pde, torch.ones_like(x_nn), create_graph=True)[0]
        y_t = autograd.grad(y_nn, x_pde, torch.ones_like(y_nn), create_graph=True)[0]
        z_t = autograd.grad(z_nn, x_pde, torch.ones_like(z_nn), create_graph=True)[0]
        loss_x = self.loss_func(x_t, sigma * (y_nn - x_nn))
        loss_y = self.loss_func(y_t, x_nn * (rho - z_nn) - y_nn)
        loss_z = self.loss_func(z_t, x_nn * y_nn - beta * z_nn)
        loss_ic = self.loss_func(y_hat[0], self.y_train_ic)
        return loss_x + loss_y + loss_z + loss_ic


if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)
    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    PINN = FCN()
    PINN.to(device)
    print(PINN)

    # 区间总数
    total_interval = 600
    # 总点数
    total_points = total_interval + 1
    # 区间长度
    h = 0.005
    x_lb = torch.tensor(0.)
    x_ub = torch.tensor(total_interval * h)

    # lorenz 方程参数
    rho = torch.tensor(15.0)
    sigma = torch.tensor(10.0)
    beta = torch.tensor(8.0 / 3.0)

    x_test = torch.linspace(x_lb, x_ub, total_points)

    # 初值
    x_train_bc = torch.tensor([[0.]])
    y_train_bc = torch.tensor([[-4., 7., 15]])

    # 配置点
    x_train_nf = x_test.unsqueeze(1)

    x_train_bc = x_train_bc.float().to(device)
    y_train_bc = y_train_bc.float().to(device)
    x_train_nf = x_train_nf.float().to(device)

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-2, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=10000,
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

        if loss.item() < 0.01:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(PINN, 'lorenz63_lstm_ic_01.pt')
    PINN.cpu()
    nn_predict = PINN(x_test[:, None]).detach().numpy()
    x_nn = nn_predict[:, [0]]
    y_nn = nn_predict[:, [1]]
    z_nn = nn_predict[:, [2]]

    fig, ax = plt.subplots()
    ax.plot(x_test, x_nn, color='r', label='x')
    ax.plot(x_test, y_nn, color='g', label='y')
    ax.plot(x_test, z_nn, color='b', label='z')
    ax.set_title('Lorenz 63')
    ax.set_xlabel('t', color='black')
    ax.set_ylabel('f(t)', color='black', rotation=0)
    ax.legend(loc='upper right')
    plt.savefig('./images/lorenz63_lstm_ic_01_2d.png')
    plt.close()
    # plt.show()

    ax = plt.axes(projection="3d")
    ax.plot(x_nn, y_nn, z_nn, color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('lorenz63')
    plt.savefig('./images/lorenz63_lstm_ic_01_3d.png')
    # plt.show()
