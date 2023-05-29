import torch
import numpy as np
from swe_pinn_01 import SWENet
import matplotlib.pyplot as plt
from matplotlib import animation


def eta_animation3D(X, Y, eta_list, frame_interval, filename):
    fig = plt.figure(figsize=(8, 8), facecolor="white")
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, eta_list[0], cmap=plt.cm.RdBu_r)

    def update_surf(num):
        ax.clear()
        surf = ax.plot_surface(X / 1000, Y / 1000, eta_list[num], cmap=plt.cm.RdBu_r)
        ax.set_title("Surface elevation $\eta(x,y,t)$ after $t={:.2f}$ hours".format(
            num * frame_interval / 3600), fontname="serif", fontsize=19, y=1.04)
        ax.set_xlabel("x [km]", fontname="serif", fontsize=14)
        ax.set_ylabel("y [km]", fontname="serif", fontsize=14)
        ax.set_zlabel("$\eta$ [m]", fontname="serif", fontsize=16)
        ax.set_xlim(X.min() / 1000, X.max() / 1000)
        ax.set_ylim(Y.min() / 1000, Y.max() / 1000)
        ax.set_zlim(-0.3, 0.7)
        plt.tight_layout()
        return surf,

    anim = animation.FuncAnimation(fig, update_surf,
                                   frames=len(eta_list), interval=10, blit=False)
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
                                         codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format(filename), writer=mpeg_writer)
    return anim  # Need to return anim object to see the animation


def velocity_animation(X, Y, u_list, v_list, frame_interval, filename):
    """Function that takes in the domain x, y (2D meshgrids) and a lists of 2D arrays
    u_list, v_list and creates an quiver animation of the velocity field (u, v). To get
    updating title one also need specify time step dt between each frame in the simulation,
    the number of time steps between each eta in eta_list and finally, a filename for video."""
    fig, ax = plt.subplots(figsize=(8, 8), facecolor="white")
    plt.title("Velocity field $\mathbf{u}(x,y)$ after 0.0 days", fontname="serif", fontsize=19)
    plt.xlabel("x [km]", fontname="serif", fontsize=16)
    plt.ylabel("y [km]", fontname="serif", fontsize=16)
    q_int = 3
    Q = ax.quiver(X[::q_int, ::q_int] / 1000.0, Y[::q_int, ::q_int] / 1000.0, u_list[0][::q_int, ::q_int],
                  v_list[0][::q_int, ::q_int],
                  scale=0.2, scale_units='inches')

    # qk = plt.quiverkey(Q, 0.9, 0.9, 0.001, "0.1 m/s", labelpos = "E", coordinates = "figure")

    # Update function for quiver animation.
    def update_quiver(num):
        u = u_list[num]
        v = v_list[num]
        ax.set_title("Velocity field $\mathbf{{u}}(x,y,t)$ after t = {:.2f} hours".format(
            num * frame_interval / 3600), fontname="serif", fontsize=19)
        Q.set_UVC(u[::q_int, ::q_int], v[::q_int, ::q_int])
        return Q,

    anim = animation.FuncAnimation(fig, update_quiver,
                                   frames=len(u_list), interval=10, blit=False)
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
                                         codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    fig.tight_layout()
    anim.save("{}.mp4".format(filename), writer=mpeg_writer)
    return anim  # Need to return anim object to see the animation


if "__main__" == __name__:
    # torch.manual_seed(123)
    # np.random.seed(123)

    model = torch.load('swe_pinn_01_gzz_100000_e-8.pt', map_location=torch.device('cpu'))
    print("model", model)

    # 空间网格设置
    N_x = 101
    N_y = 101
    N_t = 101

    x_len = 1E+6
    y_len = 1E+6
    t_len = 1E+5
    interval_len = np.array([x_len, y_len, t_len])

    x_lb = -x_len / 2
    x_ub = x_len / 2
    y_lb = -y_len / 2
    y_ub = y_len / 2
    t_lb = 0.
    t_ub = t_len

    lb = np.array([-x_len / 2, -y_len / 2, 0])
    ub = np.array([x_len / 2, y_len / 2, t_len])

    x = np.linspace(-x_len / 2, x_len / 2, N_x)
    y = np.linspace(-y_len / 2, y_len / 2, N_y)
    X, Y = np.meshgrid(x, y)

    # 构造训练数据
    x = np.linspace(x_lb, x_ub, N_x)
    y = np.linspace(y_lb, y_ub, N_y)
    X, Y = np.meshgrid(x, y)

    x_test = X.flatten()[:, None]
    y_test = Y.flatten()[:, None]

    eta_list = []
    u_list = []
    v_list = []
    for t in np.linspace(t_lb, t_ub, N_t):
        t_test = np.full((N_x * N_y, 1), t)
        z_test = np.hstack((x_test, y_test, t_test))
        z_test = torch.from_numpy(z_test).float()
        predict = model(z_test).detach().numpy()
        u = predict[:, 0].reshape(N_x, N_y)
        v = predict[:, 1].reshape(N_x, N_y)
        eta = predict[:, 2].reshape(N_x, N_y)
        print(eta.min(), eta.max())
        u_list.append(u)
        v_list.append(v)
        eta_list.append(eta)

    eta_animation3D(X, Y, eta_list, 100, "eta1")
    velocity_animation(X, Y, u_list, v_list, 100, "velocity1")
    plt.show()
