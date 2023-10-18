import numpy as np
import torch
import torch.nn as nn
import itertools
from scipy.linalg import lstsq
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
Pi = np.pi
nu = 0.01

X_min = 0.0
X_max = 8.0
T_min = 0.0
T_max = 1.0


def L_inf_error(v, axis=None):
    return np.max(np.abs(v), axis=axis)


def L_2_error(v, axis=None):
    return np.sqrt(np.sum(np.power(v, 2.0), axis=axis) / v.shape[0])


def u_real(x, t):
    """
    真解
    """
    u0 = 2 * np.cos(Pi * x + Pi / 5) + 3 / 2 * np.cos(2 * Pi * x - 3 * Pi / 5)
    u1 = 2 * np.cos(Pi * t + Pi / 5) + 3 / 2 * np.cos(2 * Pi * t - 3 * Pi / 5)

    u = u0 * u1

    return u


def f_real(x, t):
    """
    PDE的右端项f
    """
    u0 = 2 * np.cos(Pi * x + Pi / 5) + 3 / 2 * np.cos(2 * Pi * x - 3 * Pi / 5)
    u1 = 2 * np.cos(Pi * t + Pi / 5) + 3 / 2 * np.cos(2 * Pi * t - 3 * Pi / 5)

    u1t = -2 * Pi * np.sin(Pi * t + Pi / 5) - 3 * Pi * np.sin(2 * Pi * t - 3 * Pi / 5)
    ut = u0 * u1t

    u0xx = -2 * Pi * Pi * np.cos(Pi * x + Pi / 5) - 6 * Pi * Pi * np.cos(2 * Pi * x - 3 * Pi / 5)
    uxx = u0xx * u1

    f = ut - nu * uxx

    return f


def u_init(points, Nx, Qx, nt=None, qt=None):
    """
    计算整个定义域的初值
    """
    if nt is None:
        nt = 0
    if qt is None:
        qt = 0

    # 计算初值点
    point = list()

    for k in range(Nx):
        point.append(points[k][nt][:Qx, qt, :].detach().numpy().reshape((Qx, 2)))
    point = np.concatenate(point, axis=0)

    # 初值
    u_value = u_real(point[:, 0], point[:, 1]).reshape((Nx * Qx, 1))

    return u_value


def weights_init(m):
    rand_mag = 1.0
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if m.weight is not None:
            # 这里有什么深意吗？左边界为什么是0
            nn.init.uniform_(m.weight, a=0, b=rand_mag)
        if m.bias is not None:
            nn.init.uniform_(m.bias, a=-rand_mag, b=rand_mag)


class local_rep(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, M, x_max, x_min, t_max, t_min):
        super(local_rep, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = M
        self.hidden_layers = hidden_layers
        self.M = M
        # 半径的倒数
        self.a = torch.tensor([2.0 / (x_max - x_min), 2.0 / (t_max - t_min)])
        self.x_0 = torch.tensor([(x_max + x_min) / 2, (t_max + t_min) / 2])
        self.hidden_layer = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True), nn.Tanh())

    def forward(self, x):
        x = self.a * (x - self.x_0)
        x = self.hidden_layer(x)
        return x


def pre_define_rfm(Nx, Nt, M, Qx, Qt, x0, x1, t0, t1):
    models = []
    points = []
    L = x1 - x0
    tf = t1 - t0
    for k in range(Nx):
        model_for_x = []
        point_for_x = []
        x_min = L / Nx * k + x0
        x_max = L / Nx * (k + 1) + x0
        x_devide = np.linspace(x_min, x_max, Qx + 1)
        for n in range(Nt):
            t_min = tf / Nt * n + t0
            t_max = tf / Nt * (n + 1) + t0
            model = local_rep(in_features=2, out_features=1, hidden_layers=1, M=M, x_min=x_min,
                              x_max=x_max, t_min=t_min, t_max=t_max)
            model = model.apply(weights_init)
            model = model.double()
            for param in model.parameters():
                param.requires_grad = False
            model_for_x.append(model)
            t_devide = np.linspace(t_min, t_max, Qt + 1)
            # 对x和t做笛卡尔积，然后转成三维数组，本质就是这个区域的配点
            grid = np.array(list(itertools.product(x_devide, t_devide))).reshape(Qx + 1, Qt + 1, 2)
            point_for_x.append(torch.tensor(grid, requires_grad=True))
        models.append(model_for_x)
        points.append(point_for_x)
    return models, points


def cal_matrix(models, points, Nx, Nt, M, Qx, Qt, initial=None):
    # matrix define (Aw=b)
    # pde配点
    A_1 = np.zeros([Nx * Nt * Qx * Qt, Nx * Nt * M])  # u_t - lapacian u - u^2 = f(x, t) on omega [0, T]
    f_1 = np.zeros([Nx * Nt * Qx * Qt, 1])

    # 左边界配点
    A_2 = np.zeros([Nt * Qt, Nx * Nt * M])  # u(x,t) = g(x, t) on x=0
    f_2 = np.zeros([Nt * Qt, 1])

    # 右边界配点
    A_3 = np.zeros([Nt * Qt, Nx * Nt * M])  # u(x,t) = g(x, t) on x=1
    f_3 = np.zeros([Nt * Qt, 1])

    # 初值配点
    A_4 = np.zeros([Nx * Qx, Nx * Nt * M])  # u(x,t) = h(x) on t=0
    f_4 = np.zeros([Nx * Qx, 1])

    C_0x = np.zeros([(Nx - 1) * Qt * Nt, Nx * Nt * M])  # C^0 continuity on x
    f_c_0x = np.zeros([(Nx - 1) * Qt * Nt, 1])

    C_0t = np.zeros([(Nt - 1) * Qx * Nx, Nx * Nt * M])  # C^0 continuity on t
    f_c_0t = np.zeros([(Nt - 1) * Qx * Nx, 1])

    C_1x = np.zeros([(Nx - 1) * Qt * Nt, Nx * Nt * M])  # C^1 continuity on x
    f_c_1x = np.zeros([(Nx - 1) * Qt * Nt, 1])

    for k in range(Nx):
        for n in range(Nt):
            # print(k,n)
            # u_t - lapacian u - u^2 = f(x, t) on omega [0, T]
            in_ = points[k][n].detach().numpy()
            out = models[k][n](points[k][n])
            values = out.detach().numpy()
            M_begin = (k * Nt + n) * M

            # u(x, 0) = ...
            if n == 0:
                # A_4[k*Qx : (k+1)*Qx, M_begin : M_begin + M] = values[:Qx,0,:]
                if initial is None:
                    f_4[k * Qx: (k + 1) * Qx, :] = u_real(in_[:Qx, 0, 0], in_[:Qx, 0, 1]).reshape((Qx, 1))
                else:
                    f_4[k * Qx: (k + 1) * Qx, :] = initial[k * Qx: (k + 1) * Qx, :]

            # u(0,t) = ..
            if k == 0:
                # A_2[0, M_begin : M_begin + M] = values[0,:Qt,:]
                f_2[n * Qt: n * Qt + Qt, :] = u_real(in_[0, :Qt, 0], in_[0, :Qt, 1]).reshape((Qt, 1))
                # print(in_[0,:Qt,:])

            # u(1,t) = ..
            if k == Nx - 1:
                # A_5[n*Qt : n*Qt+Qt, M_begin : M_begin + M] = values[-1,:Qt,:]
                f_3[n * Qt: n * Qt + Qt, :] = u_real(in_[-1, :Qt, 0], in_[-1, :Qt, 1]).reshape((Qt, 1))
                # print(in_[-1,:Qt,:])

            # 转成二维点列，去除区域的右边界和上边界的点。然后计算方程的右端项f
            f_in = in_[:Qx, :Qt, :].reshape((-1, 2))
            f_1[k * Nt * Qx * Qt + n * Qx * Qt: k * Nt * Qx * Qt + n * Qx * Qt + Qx * Qt, :] \
                = f_real(f_in[:, 0], f_in[:, 1]).reshape(-1, 1)

            # u_t - lapacian u - u^2 = f(x, t) on omega [0, T]
            grads = []
            grads_2_xx = []
            grads_2_yy = []
            for i in range(M):
                g_1 = torch.autograd.grad(outputs=out[:, :, i], inputs=points[k][n],
                                          grad_outputs=torch.ones_like(out[:, :, i]),
                                          create_graph=True, retain_graph=True)[0]
                grads.append(g_1.squeeze().detach().numpy())

                g_2_x = torch.autograd.grad(outputs=g_1[:, :, 0], inputs=points[k][n],
                                            grad_outputs=torch.ones_like(out[:, :, i]),
                                            create_graph=True, retain_graph=True)[0]
                g_2_y = torch.autograd.grad(outputs=g_1[:, :, 1], inputs=points[k][n],
                                            grad_outputs=torch.ones_like(out[:, :, i]),
                                            create_graph=True, retain_graph=True)[0]
                grads_2_xx.append(g_2_x[:, :, 0].squeeze().detach().numpy())
                grads_2_yy.append(g_2_y[:, :, 1].squeeze().detach().numpy())

            grads = np.array(grads).swapaxes(0, 3)

            # print(values.shape,grads.shape)

            grads_t = grads[1, :Qx, :Qt, :]
            grads_t = grads_t.reshape(-1, M)
            grads_2_xx = np.array(grads_2_xx)
            grads_2_xx = grads_2_xx[:, :Qx, :Qt]
            grads_2_xx = grads_2_xx.transpose(1, 2, 0).reshape(-1, M)

            A_1[k * Nt * Qx * Qt + n * Qx * Qt: k * Nt * Qx * Qt + n * Qx * Qt + Qx * Qt,
            M_begin: M_begin + M] = grads_t - nu * grads_2_xx  # - np.power(values[:Qx, :Qt, :], 2.0).reshape(-1, M)

            # u(0,t) = ..
            if k == 0:
                A_2[n * Qt: n * Qt + Qt, M_begin: M_begin + M] = values[0, :Qt, :]
            # u(1,t) = ..
            if k == Nx - 1:
                A_3[n * Qt: n * Qt + Qt, M_begin: M_begin + M] = values[-1, :Qt, :]
            # u(x,0) = ..
            if n == 0:
                A_4[k * Qx: k * Qx + Qx, M_begin: M_begin + M] = values[:Qx, 0, :]

            # C0 continuity on t
            if n > 0:
                C_0t[(n - 1) * Qx * Nx + k * Qx: (n - 1) * Qx * Nx + k * Qx + Qx, M_begin: M_begin + M] = values[:Qx, 0,
                                                                                                          :]
            if n < Nt - 1:
                C_0t[n * Qx * Nx + k * Qx: n * Qx * Nx + k * Qx + Qx, M_begin: M_begin + M] = -values[:Qx, -1, :]

            # C0 continuity on x
            if k > 0:
                C_0x[(k - 1) * Qt * Nt + n * Qt: (k - 1) * Qt * Nt + (n + 1) * Qt, M_begin: M_begin + M] = values[0,
                                                                                                           :Qt, :]
            if k < Nx - 1:
                C_0x[k * Nt * Qt + n * Qt: k * Nt * Qt + (n + 1) * Qt, M_begin: M_begin + M] = -values[-1, :Qt, :]

            # C1 continuity on x
            if k > 0:
                C_1x[(k - 1) * Qt * Nt + n * Qt: (k - 1) * Qt * Nt + (n + 1) * Qt, M_begin: M_begin + M] = grads[0, 0,
                                                                                                           :Qt, :]
            if k < Nx - 1:
                C_1x[k * Nt * Qt + n * Qt: k * Nt * Qt + (n + 1) * Qt, M_begin: M_begin + M] = -grads[0, -1, :Qt, :]

    A = np.concatenate((A_1, A_2, A_3, A_4, C_0t, C_0x, C_1x), axis=0)
    f = np.concatenate((f_1, f_2, f_3, f_4, f_c_0t, f_c_0x, f_c_1x), axis=0)
    # print(f.shape)
    return A, f


def solve_lst_square(A, f):
    # 为什么注释掉？难道max==0的bug？这个很容易解决的啊
    # # rescaling
    # max_value = 10.0
    # for i in range(len(A)):
    #     if np.abs(A[i,:]).max()==0:
    #         print("error line : ",i)
    #         continue
    #     ratio = max_value/np.abs(A[i,:]).max()
    #     A[i,:] = A[i,:]*ratio
    #     f[i] = f[i]*ratio

    # lapack_driver参数为什么选这个？
    w, res, rnk, s = lstsq(A, f, lapack_driver="gelss")
    # print(s[0], s[-1])
    # print(np.linalg.norm(np.matmul(A, w) - f), np.linalg.norm(np.matmul(A, w) - f) / np.linalg.norm(f))

    return w


def main(Nx, Nt, M, Qx, Qt, tf=1.0, time_block=1, plot=False):
    """
    prepare models and collocation points
    """

    tlen = tf / time_block

    models, points = pre_define_rfm(Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt, x0=X_min, x1=X_max, t0=0, t1=tlen)

    initial = u_init(points=points, Nx=Nx, Qx=Qx, nt=0, qt=0)

    true_values = list()
    numerical_values = list()
    # 记录最大范数
    L_is = list()
    # 记录L2范数
    L_2s = list()

    # matrix define (Aw=b)
    A, f = cal_matrix(models, points, Nx, Nt, M, Qx, Qt, initial=initial)

    w = solve_lst_square(A, f)

    true_values_, numerical_values_, L_i, L_2 = test(models=models, \
                                                     Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt, \
                                                     w=w, \
                                                     x0=X_min, x1=X_max, t0=0, t1=T_max)

    true_values.append(true_values_)
    numerical_values.append(numerical_values_)
    L_is.append(L_i)
    L_2s.append(L_2)

    true_values = np.concatenate(true_values, axis=1)[:, ::time_block]
    numerical_values = np.concatenate(numerical_values, axis=1)[:, ::time_block]
    e_i = L_inf_error(true_values - numerical_values, axis=0).reshape(time_block, Nt, -1)
    e_2 = L_2_error(true_values - numerical_values, axis=0).reshape(time_block, Nt, -1)
    L_is = np.array(L_is)
    L_2s = np.array(L_2s)

    # visualize
    if plot:
        visualize(true_values, numerical_values, x0=X_min, x1=X_max, t0=0, t1=tf)

    return L_is, L_2s, e_i, e_2


def visualize(true_values, numerical_values, x0, x1, t0, t1, eqname="diff", mename="rfm",
              img_format="eps"):
    epsilon = np.abs(true_values - numerical_values)

    visualize_single(true_values, x0, x1, t0, t1)
    visualize_single(numerical_values, x0, x1, t0, t1)
    visualize_single(epsilon, x0, x1, t0, t1)


def visualize_single(values: np.ndarray, x0: float, x1: float, t0: float, t1: float):
    x_bins, t_bins = values.shape

    x = np.linspace(x0, x1, x_bins + 1)[:x_bins]
    y = np.linspace(t0, t1, t_bins + 1)[:t_bins]
    x, y = np.meshgrid(x, y)

    L = x1 - x0
    tf = t1 - t0

    font2 = {
        'weight': 'normal',
        'size': 24,
    }

    fig, ax = plt.subplots(figsize=[6, 8])
    fig.set_tight_layout(True)
    # ax.set_aspect('equal')
    ax.set_xlim([x0, x1])
    ax.set_ylim([t0, t1])
    ax.tick_params(labelsize=20)
    ax.set_xlabel('x', font2)
    ax.set_ylabel('t', font2)
    im = ax.pcolor(x, y, values.T, cmap='jet')
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=20)
    cbar.ax.ticklabel_format(style="sci", scilimits=(-1, 2), axis='both')
    cbar.ax.yaxis.get_offset_text().set_fontsize(12)
    cbar.update_ticks()

    plt.show()


def test(models, Nx=1, Nt=1, M=1, Mx=None, Mt=None, Qx=1, Qt=1, w=None, x0=0, x1=1, t0=0, t1=1, test_Qx=None,
         test_Qt=None, block=0):
    L = x1 - x0
    tf = t1 - t0

    if Mx is not None and Mt is not None:
        M = Mx * Mt

    epsilon = []
    true_values = []
    numerical_values = []

    if test_Qx is None:
        test_Qx = 10 * Qx
    if test_Qt is None:
        test_Qt = 10 * Qt

    tshift = block * tf

    for k in range(Nx):
        epsilon_x = []
        true_value_x = []
        numerical_value_x = []
        for n in range(Nt):
            # forward and grad
            x_min = L / Nx * k + x0
            x_max = L / Nx * (k + 1) + x0
            x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
            t_min = tf / Nt * n
            t_max = tf / Nt * (n + 1)
            t_devide = np.linspace(t_min, t_max, test_Qt + 1)[:test_Qt]
            grid = np.array(list(itertools.product(x_devide, t_devide))).reshape(test_Qx, test_Qt, 2)
            test_point = torch.tensor(grid, requires_grad=True)
            in_ = test_point.detach().numpy()
            true_value = u_real(in_[:, :, 0], in_[:, :, 1] + tshift)
            values = models[k][n](test_point).detach().numpy()
            numerical_value = np.dot(values, w[k * Nt * M + n * M: k * Nt * M + n * M + M, :]).reshape(test_Qx, test_Qt)
            e = np.abs(true_value - numerical_value)
            true_value_x.append(true_value)
            numerical_value_x.append(numerical_value)
            epsilon_x.append(e)
        epsilon_x = np.concatenate(epsilon_x, axis=1)
        epsilon.append(epsilon_x)
        true_value_x = np.concatenate(true_value_x, axis=1)
        true_values.append(true_value_x)
        numerical_value_x = np.concatenate(numerical_value_x, axis=1)
        numerical_values.append(numerical_value_x)
    epsilon = np.concatenate(epsilon, axis=0)
    true_values = np.concatenate(true_values, axis=0)
    numerical_values = np.concatenate(numerical_values, axis=0)
    e = epsilon.reshape((-1, 1))
    L_i = L_inf_error(e)
    L_2 = L_2_error(e)
    print('********************* ERROR *********************')
    print(f"Block={block:d},t0={tshift:.2f},t1={tshift + tf:.2f}")
    if Mx is not None and Mt is not None:
        print('Nx={:d},Nt={:d},Mx={:d},Mt={:d},Qx={:d},Qt={:d}'.format(Nx, Nt, Mx, Mt, Qx, Qt))
    else:
        print('Nx={:d},Nt={:d},M={:d},Qx={:d},Qt={:d}'.format(Nx, Nt, M, Qx, Qt))
    print('L_inf={:.2e}'.format(L_i), 'L_2={:.2e}'.format(L_2))
    print("边值条件误差")
    print("{:.2e} {:.2e}".format(max(epsilon[0, :]), max(epsilon[-1, :])))
    print("初值、终值误差")
    print("{:.2e} {:.2e}".format(max(epsilon[:, 0]), max(epsilon[:, -1])))
    # np.save('./epsilon_psi2.npy',epsilon)

    return true_values, numerical_values, L_i, L_2


if __name__ == '__main__':
    # 时间块
    time_block = 1
    # x维度划分的区间数
    Nx = 5
    # t维度划分的区间数
    Nt = 2
    # 每个局部局域的特征函数数量
    M = 300
    # x维度每个区间的配点数，Qx+1
    Qx = 30
    # t维度每个区间的配点数，Qt+1
    Qt = 30
    # time final
    tf = 1.0
    recursive_times = 1

    L_i_res = list()
    L_2_res = list()
    e_i_res = list()
    e_2_res = list()
    result = []

    for k in range(recursive_times):
        plot = (k == 0)

        L_is, L_2s, e_i, e_2 = main(Nx=Nx, Nt=Nt, M=M, tf=tf, time_block=time_block, Qx=Qx, Qt=Qt, plot=plot)

        L_i_res.append(L_is)
        L_2_res.append(L_2s)
        e_i_res.append(np.stack(e_i, axis=0))
        e_2_res.append(np.stack(e_2, axis=0))

        L_i = L_inf_error(L_is)
        L_2 = L_2_error(L_2s)

        result.append([L_i, L_2])

    L_i_res = np.array(L_i_res)
    L_2_res = np.array(L_2_res)
    e_i_res = np.array(e_i_res).reshape(recursive_times, time_block, Nt, -1)
    e_2_res = np.array(e_2_res).reshape(recursive_times, time_block, Nt, -1)

    result = np.array(result)
