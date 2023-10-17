import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from swe_1d_don_tbc_obs_02 import SWENet

if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    # 测试数据集取后100条
    test_dataset = np.load('swe_1d_rbf_sample_1000_l=0.2.npy')[-100:]
    dataset_length = len(test_dataset)
    seq_idx = np.random.randint(dataset_length)
    print("seq_idx :", seq_idx)

    obs_index = [10, 20, 30, 40, 50]

    seq = test_dataset[seq_idx]
    ic = seq[0]
    obs_h = seq[obs_index, 50]
    obs_v = seq[obs_index, 151]
    o_test = np.concatenate((ic[:101], obs_h, obs_v))

    # 这里先把数值解确定下
    h_real = seq[:51, :101]

    N_x = 101
    N_t = 51

    x = np.linspace(0, 1, N_x)
    t = np.linspace(0, 0.5, N_t)

    X, T = np.meshgrid(x, t)
    x_test = X.flatten()[:, None]
    t_test = T.flatten()[:, None]
    y_test = np.hstack((x_test, t_test))

    # 拼接输入
    o_test = np.repeat(o_test[None, :], len(y_test), axis=0)
    z_test = np.hstack((y_test, o_test))
    z_test = torch.from_numpy(z_test).float()

    model = torch.load('swe_1d_don_tbc_obs_02_5e-6.pt', map_location=torch.device('cpu'))
    print("model", model)

    predict = model(z_test).detach().numpy()
    h_hat = predict[:, 0].reshape(-1, N_x)

    # 创建图表对象
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # 初始化曲线
    line1, = ax.plot([], [], color="red", linestyle='-', label="t=")
    line2, = ax.plot([], [], color="blue", linestyle='--', label="t=")


    # 更新函数
    def update(i):
        line1.set_data(x, h_hat[i])
        line2.set_data(x, h_real[i])
        line1.set_label("t=" + str(round(i * 0.01, 2)))
        line2.set_label("err=" + str(round(np.linalg.norm(h_real[i] - h_hat[i]) / np.linalg.norm(h_real[i]), 5)))
        ax.legend()
        return line1, line2,


    # 创建动画
    anim = animation.FuncAnimation(fig, update, frames=len(h_hat), interval=100)
    # 保存动画
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
                                         codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format("swe_1d_don_tbc_obs_02"), writer=mpeg_writer)

    plt.show()
