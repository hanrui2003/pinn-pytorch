import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation

if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    torch.set_default_dtype(torch.float)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    train_data = np.load('swe_1d_rbf_sample_1000_l=0.2.npy')

    # 随机选择一个，绘图，测试下数据的有效性
    index = np.random.randint(1000)
    u = train_data[index]
    h = u[:, 0:101]
    # 创建图表对象
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # 初始化曲线
    line1, = ax.plot([], [], color="red", linestyle='-', label="t=")

    x_nn = np.linspace(0, 1, 101)


    # 更新函数
    def update(i):
        line1.set_data(x_nn, h[i])
        line1.set_label("t=" + str(round(i * 0.01, 2)))
        ax.legend()
        return line1,


    # 创建动画
    anim = animation.FuncAnimation(fig, update, frames=len(h), interval=100)
    # 保存动画
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
                                         codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format("swe_1d_rbf_sample_4"), writer=mpeg_writer)

    plt.show()
