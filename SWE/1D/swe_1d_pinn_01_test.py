import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from swe_1d_pinn_01 import SWENet

if "__main__" == __name__:
    torch.manual_seed(123)
    np.random.seed(123)

    model = torch.load('swe_1d_pinn_01_e-6.pt', map_location=torch.device('cpu'))
    print("model", model)

    x = np.linspace(0, 1, 101)
    t = np.linspace(0, 1, 101)
    X, T = np.meshgrid(x, t)
    x_test = X.flatten()[:, None]
    t_test = T.flatten()[:, None]
    y_test = np.hstack((x_test, t_test))
    y_test = torch.from_numpy(y_test).float()
    predict = model(y_test).detach().numpy()
    h_hat = predict[:, 0].reshape(101, 101)

    # 创建图表对象
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # 初始化曲线
    line, = ax.plot([], [], label="t=")


    # 更新函数
    def update(i):
        line.set_data(x, h_hat[i])
        line.set_label("t=" + str(round(i * 0.01, 2)))
        ax.legend()
        return line,


    # 创建动画
    anim = animation.FuncAnimation(fig, update, frames=len(h_hat), interval=100)
    # 保存动画
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
                                         codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format("swe-1d"), writer=mpeg_writer)

    plt.show()

    print()
