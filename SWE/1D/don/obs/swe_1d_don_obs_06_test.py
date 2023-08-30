import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.integrate import solve_ivp
from swe_1d_don_obs_05 import SWENet

if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    sample_data = np.load('swe_1d_rbf_sample_1.npy')
    seq_len = len(sample_data)
    seq_idx = np.random.randint(seq_len)
    ic_idx = np.random.randint(500)
    print("seq_idx :", seq_idx, "ic_idx :", ic_idx)

    delta_index = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    obs_index = ic_idx + delta_index

    seq = sample_data[seq_idx]
    ic = seq[ic_idx]
    obs_h = seq[obs_index, 50]
    obs_v = seq[obs_index, 151]
    o_test = np.concatenate((ic[:101], obs_h, ic[101:], obs_v))
    o_test = torch.from_numpy(o_test).float()

    # 这里先把数值解确定下
    h_real = seq[ic_idx:ic_idx + 101, :101]

    N_x = 101
    N_t = 101

    x = np.linspace(0, 1, N_x)
    t = np.linspace(0, 1, N_t)

    X, T = np.meshgrid(x, t)
    x_test = X.flatten()[:, None]
    t_test = T.flatten()[:, None]
    y_test = np.hstack((x_test, t_test))
    y_test = torch.from_numpy(y_test).float()

    model = torch.load('swe_1d_don_obs_06_5e-5.pt', map_location=torch.device('cpu'))
    print("model", model)

    predict = model(o_test, y_test).detach().numpy()
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
    anim.save("{}.mp4".format("swe_1d_don_obs_06"), writer=mpeg_writer)

    plt.show()
