import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from fdm import *

torch.set_default_dtype(torch.float64)

R_m = 1.0


def weights_init(m):
    """
    网络参数初始化，采用均匀分布。
    :param m:
    :return:
    """
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.uniform_(m.weight, a=-R_m, b=R_m)
        nn.init.uniform_(m.bias, a=-R_m, b=R_m)


# network definition
class Network(nn.Module):
    """
    一个简单的全连接网络，单隐层
    输入是d维，输出一维，中间一个隐藏层，维度是M
    """

    def __init__(self, d, M):
        super(Network, self).__init__()
        self.fc_layer = nn.Sequential(nn.Linear(d, M, bias=True), nn.Tanh())
        self.output_layer = nn.Linear(M, 1, bias=False)

    def forward(self, x):
        h = self.fc_layer(x)
        out = self.output_layer(h)
        return out


# loss definition
class Loss:
    def __init__(self, net, left_boundary, right_boundary):
        self.net = net
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

    def sample(self, N):
        """
        就是返回离散点（配点），x是除了边界的离散点，也就是内部配点
        N: 离散的区间数
        """
        # x是除了边界的离散点，也就是内部配点
        x = torch.tensor(np.float64(np.linspace(self.left_boundary, self.right_boundary, N + 1)[1:N]),
                         requires_grad=True).reshape(-1, 1)
        x_boundary_left = torch.ones([1], requires_grad=True) * self.left_boundary
        x_boundary_right = torch.ones([1], requires_grad=True) * self.right_boundary
        return x, x_boundary_left, x_boundary_right

    def loss_func(self, N):
        """
        两个边值是按带标签的计算损失，中间的点算物理损失。
        :param N:
        :return:
        """
        # helmholtz equation
        x, x_boundary_left, x_boundary_right = self.sample(N)
        x = Variable(x, requires_grad=True)
        y = self.net(x)
        dx = torch.autograd.grad(y, x, grad_outputs=torch.ones_like(self.net(x)), create_graph=True)[0].reshape(-1, 1)
        dxx = torch.autograd.grad(dx, x, grad_outputs=torch.ones_like(dx), create_graph=True)[0].reshape(-1, 1)

        # f是已知的，就是fdm.py里面的f(x)，只不过这里不用numpy，用的torch的张量，所以要重写
        # 这里有点问题，写成了 d2u_dx2(x) - lamb * u(x) ， 应该是d2u_dx2(x) + lamb * u(x) ，不过下面的计算损失也是犯了同样的错误，所以抵消了。
        f = -AA * (aa * aa + bb * bb) * torch.sin(bb * (x + 0.05)) * torch.cos(aa * (x + 0.05)) \
            - 2.0 * AA * aa * bb * torch.cos(bb * (x + 0.05)) * torch.sin(aa * (x + 0.05)) \
            - lamb * (AA * torch.sin(bb * (x + 0.05)) * torch.cos(aa * (x + 0.05)) + 2.0)
        # 这里和上面一样，加写成了减。
        diff_error = (dxx - lamb * self.net(x) - f.reshape(-1, 1)) ** 2

        # boundary condition
        bd_left_error = (self.net(x_boundary_left) - (AA * torch.sin(bb * (x_boundary_left + 0.05)) * torch.cos(
            aa * (x_boundary_left + 0.05)) + 2.0)) ** 2
        bd_right_error = (self.net(x_boundary_right) - (AA * torch.sin(bb * (x_boundary_right + 0.05)) * torch.cos(
            aa * (x_boundary_right + 0.05)) + 2.0)) ** 2

        # 把边界点的损失和物理点的损失算均值
        return torch.mean(diff_error + bd_left_error + bd_right_error)


# training process
class Train():
    def __init__(self, net, loss, N):
        self.errors = []
        self.N = N
        self.net = net
        self.model = loss

    def train(self, epoch, lr):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()), lr)
        avg_loss = 0
        for e in range(epoch):
            optimizer.zero_grad()
            loss = self.model.loss_func(self.N)
            avg_loss = avg_loss + float(loss.item())
            loss.backward()
            optimizer.step()
            if e % 50 == 49:
                loss = avg_loss / 50
                # print("Epoch {} - lr {} -  loss: {}".format(e, lr, loss))
                avg_loss = 0
                error = self.model.loss_func(self.N)
                self.errors.append(error.detach())


# test process
def Test(net, N):
    # PINN testing process
    points = np.linspace(left_boundary, right_boundary, N + 1, dtype=np.float64)[:N]
    true_value = u(points).reshape([-1, 1])
    numerical_value = net(torch.tensor(points, requires_grad=False).reshape([-1, 1])).detach().cpu().numpy()
    epsilon = true_value - numerical_value
    epsilon = np.abs(epsilon)
    error_inf = epsilon.max()
    error_l2 = math.sqrt(8 * sum(epsilon * epsilon) / len(epsilon))
    print('L_infty=', error_inf, 'L_2=', error_l2)
    return (error_l2)


if __name__ == "__main__":
    PINN_Error = np.zeros([5, 3])
    for i in range(5):
        time_begin = time.time()
        N = int(100 * 2 ** i)  # number of collocation points
        M = int(100 * 2 ** i)  # number of basis

        # PINN model define and initialization
        net = Network(1, M)
        net.fc_layer = net.fc_layer.apply(weights_init)

        # PINN training process
        FREEZE = False
        if FREEZE == True:
            net.fc_layer.requires_grad_(False)
            loss = Loss(net, left_boundary, right_boundary)
            train = Train(net, loss, N)
            train.train(epoch=10 ** 2, lr=0.001)
            train.train(epoch=4 * 10 ** 2, lr=0.0001)
        else:
            loss = Loss(net, left_boundary, right_boundary)
            train = Train(net, loss, N)
            train.train(epoch=5 * 10 ** 3, lr=0.1)
            train.train(epoch=10 ** 4, lr=0.01)
            train.train(epoch=10 ** 4, lr=0.001)
            train.train(epoch=2 * 10 ** 4, lr=0.0001)

        # PINN testing process
        print('***********************')
        print("N = M =", N)
        PINN_Error[i, 0] = N
        PINN_Error[i, 1] = Test(net, N)
        time_end = time.time()
        PINN_Error[i, 2] = time_end - time_begin

    error_plot([FDM_Error, PINN_Error])
    time_plot([FDM_Error, PINN_Error])
