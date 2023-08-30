import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from swe_1d_pinn_obs_01 import SWENet

if "__main__" == __name__:
    torch.manual_seed(123)
    np.random.seed(123)

    sample_data = np.load('swe_1d_rbf_sample_1.npy')

    model = torch.load('swe_1d_pinn_obs_01_e-6.pt', map_location=torch.device('cpu'))
    print("model", model)

    N_x = 101
    N_t = 101

    x = np.linspace(0, 1, N_x)
    t = np.linspace(0, 1, N_t)

    X, T = np.meshgrid(x, t)
    x_test = X.flatten()[:, None]
    t_test = T.flatten()[:, None]
    y_test = np.hstack((x_test, t_test))
    y_test = torch.from_numpy(y_test).float()
    predict = model(y_test).detach().numpy()
    h_hat = predict[:, 0].reshape(-1, N_x)

    # 数值解， 直接取样本数据即可
    h_real = sample_data[0][:101, :101]

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
    anim.save("{}.mp4".format("swe_1d_pinn_obs_01"), writer=mpeg_writer)

    plt.show()

    print()
