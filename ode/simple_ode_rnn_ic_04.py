import torch
import torch.nn as nn
import torch.autograd as autograd
import matplotlib.pyplot as plt
import numpy as np
from pyDOE import lhs


class FCN(nn.Module):
    def __init__(self, x_lb, x_ub, pre_input_size=1, pre_hidden_size=32, pre_num_layers=2,
                 post_input_size=32, post_hidden_size=1, post_num_layers=1):
        super().__init__()
        self.x_lb = x_lb
        self.x_ub = x_ub

        self.loss_func = nn.MSELoss(reduction='mean')
        # 默认的 h_0 ，拟合的结果，开始会抖，虽然抖的不是很厉害

        self.prev_h_0 = nn.Parameter(torch.randn(pre_num_layers, pre_hidden_size))
        self.post_h_0 = torch.tensor([[np.sin(-0.005)]]).float()
        # self.post_h_0 = nn.Parameter(torch.randn(post_num_layers, post_hidden_size))
        print("prev_h_0 : ", self.prev_h_0)
        print("post_h_0 : ", self.post_h_0)

        self.prev_rnn = nn.RNN(input_size=pre_input_size, hidden_size=pre_hidden_size, num_layers=pre_num_layers,
                               batch_first=True)
        self.post_rnn = nn.RNN(input_size=post_input_size, hidden_size=post_hidden_size, num_layers=post_num_layers,
                               batch_first=True)

    def forward(self, x):
        # 正规化
        x = (x - self.x_lb) / (self.x_ub - self.x_lb)
        x, _ = self.prev_rnn(x, self.prev_h_0)
        x, _ = self.post_rnn(x, self.post_h_0)
        return x

    def loss(self, x_train, y_train):
        y_hat = self.forward(x_train)
        return self.loss_func(y_hat, y_train)


if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)
    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    cpu_device = torch.device('cpu')
    # 区间总数
    total_interval = 600
    # 总点数
    total_points = total_interval + 1
    # 区间长度
    h = 0.005
    x_lb = torch.tensor(0.)
    x_ub = torch.tensor(total_interval * h)

    PINN = FCN(x_lb, x_ub)
    PINN.to(device)
    print(PINN)

    x_test = torch.linspace(x_lb, x_ub, total_points)

    # 配置点
    x_train = x_test.unsqueeze(1)
    y_train = torch.sin(x_test).unsqueeze(1)

    x_train = x_train.float().to(device)
    y_train = y_train.float().to(device)

    optimizer = torch.optim.Adam(PINN.parameters(), lr=1e-3, amsgrad=False)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-6, mode='min', factor=0.5,
                                                           patience=40000,
                                                           verbose=True)
    epoch = 0
    while True:
        epoch += 1
        with torch.backends.cudnn.flags(enabled=False):
            loss = PINN.loss(x_train, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss)

        if epoch % 100 == 0:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())

        if loss.item() < 0.0001:
            print('epoch :', epoch, 'lr :', optimizer.param_groups[0]['lr'], 'loss :', loss.item())
            break

    torch.save(PINN, 'simple_ode_rnn_ic_04.pt')
    PINN.cpu()
    print("prev_h_0 : ", PINN.prev_h_0)
    print("post_h_0 : ", PINN.post_h_0)

    nn_predict = PINN(x_test[:, None]).detach().numpy()
    # x_nn = nn_predict[:, [0]]
    # y_nn = nn_predict[:, [1]]
    # z_nn = nn_predict[:, [2]]

    print(nn_predict[:10])

    fig, ax = plt.subplots()
    ax.plot(x_test, nn_predict, 'r-', linewidth=2, label='x')
    ax.plot(x_test, y_train, 'g--', linewidth=2, label='truth')
    # ax.plot(x_test, z_nn, color='b', label='z')
    ax.set_title('$sin x$')
    ax.set_xlabel('t', color='black')
    ax.set_ylabel('f(t)', color='black', rotation=0)
    ax.legend(loc='upper right')
    plt.savefig('./figure/simple_ode_rnn_ic_04.png')
    plt.show()
