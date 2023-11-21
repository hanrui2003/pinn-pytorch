import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp
from swe_1d_don_tbc_da_05 import SWENet


def plot(X, T, U_true, U_numerical, U_nn):
    """
    同时绘制PDE的数值解和神经网络解，上面数值解，下面神经网络解。
    """
    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(15, 8))

    ax1 = fig.add_subplot(231)
    ax2 = fig.add_subplot(232)
    ax3 = fig.add_subplot(233)
    ax4 = fig.add_subplot(234, projection='3d')
    ax5 = fig.add_subplot(235, projection='3d')
    ax6 = fig.add_subplot(236, projection='3d')

    cp1 = ax1.contourf(T, X, U_true, 20, cmap="rainbow")
    fig.colorbar(cp1, ax=ax1)
    ax1.set_title('true(x,t)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')

    cp2 = ax2.contourf(T, X, U_numerical, 20, cmap="rainbow")
    fig.colorbar(cp2, ax=ax2)
    ax2.set_title('numerical(x,t)')
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')

    cp3 = ax3.contourf(T, X, U_nn, 20, cmap="rainbow")
    fig.colorbar(cp3, ax=ax3)
    ax3.set_title('nn(x,t)')
    ax3.set_xlabel('t')
    ax3.set_ylabel('x')

    ax4.plot_surface(T, X, U_true, cmap="rainbow")
    ax4.set_xlabel('t')
    ax4.set_ylabel('x')
    ax4.set_zlabel('true(x,t)')

    ax5.plot_surface(T, X, U_numerical, cmap="rainbow")
    ax5.set_xlabel('t')
    ax5.set_ylabel('x')
    ax5.set_zlabel('numerical(x,t)')

    ax6.plot_surface(T, X, U_nn, cmap="rainbow")
    ax6.set_xlabel('t')
    ax6.set_ylabel('x')
    ax6.set_zlabel('nn(x,t)')

    plt.show()


def dudt(u_n, dx):
    # 波速
    c = 0.2
    # 重力加速度
    g = 1.0
    length = len(u_n)
    half_len = length // 2
    curr_dhdt = np.zeros(half_len)
    curr_dmdt = np.zeros(half_len)
    curr_h = u_n[:half_len]
    curr_m = u_n[half_len:]
    for i in range(half_len):
        if i == 0:
            curr_dhdt[i] = (-curr_m[i] - curr_m[i + 1] + c * (curr_h[i] - 2 * curr_h[i] + curr_h[i + 1])) / (2 * dx)
            curr_dmdt[i] = ((curr_m[i] ** 2 / curr_h[i] - curr_m[i + 1] ** 2 / curr_h[i + 1]) + (
                    curr_h[i] ** 2 - curr_h[i + 1] ** 2) * g / 2 + c * (
                                    -curr_m[i] - 2 * curr_m[i] + curr_m[i + 1])) / (2 * dx)
        elif i == half_len - 1:
            curr_dhdt[i] = (curr_m[i - 1] + curr_m[i] + c * (curr_h[i - 1] - 2 * curr_h[i] + curr_h[i])) / (2 * dx)
            curr_dmdt[i] = ((curr_m[i - 1] ** 2 / curr_h[i - 1] - curr_m[i] ** 2 / curr_h[i]) + (
                    curr_h[i - 1] ** 2 - curr_h[i] ** 2) * g / 2 + c * (
                                    curr_m[i - 1] - 2 * curr_m[i] - curr_m[i])) / (2 * dx)
        else:
            curr_dhdt[i] = (curr_m[i - 1] - curr_m[i + 1] + c * (curr_h[i - 1] - 2 * curr_h[i] + curr_h[i + 1])) / (
                    2 * dx)
            curr_dmdt[i] = ((curr_m[i - 1] ** 2 / curr_h[i - 1] - curr_m[i + 1] ** 2 / curr_h[i + 1]) + (
                    curr_h[i - 1] ** 2 - curr_h[i + 1] ** 2) * g / 2 + c * (
                                    curr_m[i - 1] - 2 * curr_m[i] + curr_m[i + 1])) / (2 * dx)

    return np.hstack((curr_dhdt, curr_dmdt))


def rk4(u0, n, dx, dt):
    """

    :param u0: 初值
    :param n: 迭代次数
    :param dt: 时间步长
    :return: 数值解
    """
    u = np.zeros((n + 1, len(u0)))
    u[0] = u0
    for i in range(n):
        k1 = dudt(u[i], dx)
        k2 = dudt(u[i] + 0.5 * dt * k1, dx)
        k3 = dudt(u[i] + 0.5 * dt * k2, dx)
        k4 = dudt(u[i] + dt * k3, dx)
        u[i + 1] = u[i] + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return u


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    # 后100个作为测试集
    total_data = np.load('swe_1d_rbf_sample_4000_l_0.2_t_1.npy')
    test_dataset = total_data[900:1000]
    noise_dataset = total_data[1900:2000]

    dataset_length = len(test_dataset)
    seq_idx = np.random.randint(dataset_length)
    print("seq_idx :", seq_idx)

    obs_index = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    seq = test_dataset[seq_idx]
    ic = seq[0] + 0.1 * noise_dataset[seq_idx][0]
    obs_h_1 = seq[obs_index, 20]
    obs_v_1 = seq[obs_index, 121]
    obs_h_2 = seq[obs_index, 40]
    obs_v_2 = seq[obs_index, 141]
    obs_h_3 = seq[obs_index, 60]
    obs_v_3 = seq[obs_index, 161]
    obs_h_4 = seq[obs_index, 80]
    obs_v_4 = seq[obs_index, 181]
    branch_test = np.concatenate((ic[:101], obs_h_1, obs_h_2, obs_h_3, obs_h_4, obs_v_1, obs_v_2, obs_v_3, obs_v_4))

    # 真解
    U_true = seq[:, :101]

    # 计算传统的数值解
    N_x_numerical = 101
    N_t_numerical = 101

    x_numerical = np.linspace(0, 1, N_x_numerical)
    t_numerical = np.linspace(0, 1, N_t_numerical)
    dx = x_numerical[1] - x_numerical[0]
    dt = t_numerical[1] - t_numerical[0]
    U_numerical = rk4(ic, N_t_numerical - 1, dx, dt)[:, :101]

    # 神经网络解
    N_x_nn = 101
    N_t_nn = 101

    x_nn = np.linspace(0, 1, N_x_nn)
    t_nn = np.linspace(0, 1, N_t_nn)

    X, T = np.meshgrid(x_nn, t_nn)
    x_test = X.flatten()[:, None]
    t_test = T.flatten()[:, None]
    y_test = np.hstack((x_test, t_test))

    # 拼接输入
    branch_test = np.repeat(branch_test[None, :], len(y_test), axis=0)
    z_test = np.hstack((y_test, branch_test))
    z_test = torch.from_numpy(z_test).float()

    # 构建网络网络结构
    layers = [183, 256, 256, 256, 256, 2]
    model = SWENet(layers)
    model.load_state_dict(torch.load('swe_1d_don_tbc_da_05_5e-5.pt', map_location=torch.device('cpu')))
    model.eval()
    print("model", model)

    predict = model(z_test).detach().numpy()
    U_nn = predict[:, 0].reshape(-1, N_x_nn)

    epsilon_numerical = np.abs(U_true - U_numerical)
    epsilon_nn = np.abs(U_true - U_nn)

    L_inf_numerical = np.max(epsilon_numerical)
    L_2_numerical = np.sqrt(np.sum(epsilon_numerical ** 2) / epsilon_numerical.size)
    L_rel_numerical = np.linalg.norm(epsilon_numerical) / np.linalg.norm(U_true)
    print('L_inf_numerical :', L_inf_numerical, ' , L_2_numerical', L_2_numerical, ' , L_rel_numerical : ',
          L_rel_numerical)

    L_inf_nn = np.max(epsilon_nn)
    L_2_nn = np.sqrt(np.sum(epsilon_nn ** 2) / epsilon_nn.size)
    L_rel_nn = np.linalg.norm(epsilon_nn) / np.linalg.norm(U_true)
    print('L_inf_nn :', L_inf_nn, ' , L_2_nn', L_2_nn, ' , L_rel_nn : ', L_rel_nn)

    # plot(X, T, U_true, U_numerical, U_nn)

    # # 创建图表对象
    # fig, ax = plt.subplots()
    # ax.set_xlim(0, 1)
    # ax.set_ylim(0, 0.3)
    # plt.grid(True, color='gray', linestyle='--', linewidth=0.5)
    #
    # # 初始化曲线
    # line1, = ax.plot([], [], color="red", linestyle='-', label="t=")
    # line2, = ax.plot([], [], color="blue", linestyle='--', label="t=")
    #
    #
    # # 更新函数
    # def update(i):
    #     line1.set_data(x, h_hat[i])
    #     line2.set_data(x, h_real[i])
    #     line1.set_label("t=" + str(round(i * 0.01, 2)))
    #     line2.set_label("err=" + str(round(np.linalg.norm(h_real[i] - h_hat[i]) / np.linalg.norm(h_real[i]), 5)))
    #     ax.legend()
    #     return line1, line2,
    #
    #
    # # 创建动画
    # anim = animation.FuncAnimation(fig, update, frames=len(ns), interval=100)
    # # 保存动画
    # mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
    #                                      codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    # anim.save("{}.mp4".format("swe_1d_don_tbc_da_01"), writer=mpeg_writer)
    #
    # plt.show()
