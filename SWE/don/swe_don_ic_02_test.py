import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import animation
from swe_don_ic_02 import SWENet


def plot(X1, T1, U1):
    """
    同时绘制PDE的数值解和神经网络解，上面数值解，下面神经网络解。
    """
    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(12, 8))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    cp1 = ax1.contourf(T1, X1, U1, 20, cmap="rainbow")
    fig.colorbar(cp1, ax=ax1)
    ax1.set_title('u(x,t)')
    ax1.set_xlabel('t')
    ax1.set_ylabel('x')

    ax2.plot_surface(T1, X1, U1, cmap="rainbow")
    ax2.set_xlabel('t')
    ax2.set_ylabel('x')
    ax2.set_zlabel('u(x,t)')

    plt.show()


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
        # ax.set_xlim(X.min() / 1000, X.max() / 1000)
        # ax.set_ylim(Y.min() / 1000, Y.max() / 1000)
        ax.set_zlim(0., 0.02)
        plt.tight_layout()
        return surf,

    anim = animation.FuncAnimation(fig, update_surf, frames=len(eta_list), interval=10, blit=False)
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format(filename), writer=mpeg_writer)
    return anim  # Need to return anim object to see the animation


if "__main__" == __name__:
    torch.manual_seed(123)
    np.random.seed(123)

    model = torch.load('swe_don_ic_02_gzz_e-6.pt', map_location=torch.device('cpu'))
    print("model", model)

    N_x = 101
    N_y = 101
    N_t = 101

    x_lb = 0.
    x_ub = 1.
    y_lb = 0.
    y_ub = 1.
    t_lb = 0.
    t_ub = 1.

    # 构造训练数据
    mu = np.random.uniform(low=[0.05, 0.05], high=[0.15, 0.15], size=(1, 2))[0]
    x = np.linspace(x_lb, x_ub, N_x)
    y = np.linspace(y_lb, y_ub, N_y)
    X, Y = np.meshgrid(x, y)

    sigma = 0.02
    eta_test = np.exp(-((X - mu[0]) ** 2 + (Y - mu[1]) ** 2) / (sigma ** 2))
    eta_test = eta_test[:20, :20].flatten()
    eta_test = np.tile(eta_test, (N_x * N_y, 1))
    eta_test = torch.from_numpy(eta_test).float()

    x_test = X.flatten()[:, None]
    y_test = Y.flatten()[:, None]

    eta_list = []
    for t in np.linspace(t_lb, t_ub, N_y):
        t_test = np.full((N_x * N_y, 1), t)
        z_test = np.hstack((x_test, y_test, t_test))
        z_test = torch.from_numpy(z_test).float()
        predict = model(eta_test, z_test).detach().numpy()
        eta = predict[:, 2].reshape(N_x, N_y)
        print(eta.min(), eta.max())
        eta_list.append(eta)

    eta_animation3D(X, Y, eta_list, 100, "eta")
    plt.show()
