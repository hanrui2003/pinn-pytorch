import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np


def eta_animation3D(X, Y, eta_list, frame_interval, filename):
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, eta_list[0], cmap=plt.cm.RdBu_r)

    def update_surf(num):
        ax.clear()
        surf = ax.plot_surface(X, Y, eta_list[num], cmap=plt.cm.RdBu_r)
        ax.set_title("Surface elevation $\eta(x,y,t)$ after $t={:.2f}$ hours".format(
            num * frame_interval / 3600), fontname="serif", fontsize=19, y=1.04)
        ax.set_xlabel("x [km]", fontname="serif", fontsize=14)
        ax.set_ylabel("y [km]", fontname="serif", fontsize=14)
        ax.set_zlabel("$\eta$ [m]", fontname="serif", fontsize=16)
        ax.set_zlim(-0.3, 1.)
        plt.tight_layout()
        return surf,

    anim = animation.FuncAnimation(fig, update_surf, frames=len(eta_list), interval=10, blit=False)
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format(filename), writer=mpeg_writer)
    return anim  # Need to return anim object to see the animation


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    N_x = 101
    N_y = 101

    x_lb = 0.
    x_ub = 1.
    y_lb = 0.
    y_ub = 1.
    t_lb = 0.
    t_ub = 1.

    eta_train_count = 500
    physics_train_count = 1000

    # 构造训练数据
    mu = np.random.uniform(low=[0.05, 0.05], high=[0.15, 0.15], size=(eta_train_count, 2))
    x = np.linspace(x_lb, x_ub, N_x)
    y = np.linspace(y_lb, y_ub, N_y)
    X, Y = np.meshgrid(x, y)

    x_test = X.flatten()[:, None]
    y_test = Y.flatten()[:, None]

    eta_train_raw = []
    sigma = 0.02
    for m in mu:
        eta = np.exp(-((X - m[0]) ** 2 + (Y - m[1]) ** 2) / (sigma ** 2))
        eta_train_raw.append(eta)

    eta_animation3D(X, Y, eta_train_raw, 100, "eta")
    plt.show()
