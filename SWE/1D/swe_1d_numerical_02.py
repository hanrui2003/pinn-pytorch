import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation


# initial height profile
def u0(x):
    return 0.1 + 0.1 * np.exp(-64 * (x - 0.25) ** 2)


def f(H, M, i):
    return M[i], M[i] * M[i] / H[i] + (g / 2) * H[i] * H[i]


def flux(H, M, L, R):
    if L == -1:
        return 0, f(H, M, R)[1]
    if R == div:
        return 0, f(H, M, L)[1]
    fl = f(H, M, L)
    fr = f(H, M, R)
    d = 0.2
    return (fl[0] + fr[0]) / 2 + d * (H[L] - H[R]) / 2, (fl[1] + fr[1]) / 2 + d * (M[L] - M[R]) / 2


def dUdt(u_n):
    curr_dhdt = []
    curr_dmdt = []
    curr_h = u_n[:div]
    curr_m = u_n[div:]
    for i in range(div):
        flux_l = flux(curr_h, curr_m, i - 1, i)
        flux_r = flux(curr_h, curr_m, i, i + 1)
        curr_dhdt.append((flux_l[0] - flux_r[0]) / dx)
        curr_dmdt.append((flux_l[1] - flux_r[1]) / dx)
    return curr_dhdt + curr_dmdt


def rk4(f, h):
    for n in range(500):
        k1 = np.asarray(f(u[n]))
        k2 = np.asarray(f(u[n] + 0.5 * h * k1))
        k3 = np.asarray(f(u[n] + 0.5 * h * k2))
        k4 = np.asarray(f(u[n] + h * k3))
        u[n + 1] = u[n] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


if "__main__" == __name__:
    # mesh parameter
    L = 1
    div = 100
    dx = L / div

    # time parameter
    t0 = 0.0
    T = 5

    # physics constants
    g = 1.0

    # initial height vector
    # ini_h = np.array([u0((i + 0.5) * dx) for i in range(div)])
    ini_h_left = 0.2 * np.ones(20)
    ini_h_right = 0.1 * np.ones(80)
    ini_h = np.hstack((ini_h_left, ini_h_right))
    # initial velocity vector
    ini_m = np.array([0 for i in range(div)])

    u = np.zeros((501, 200))
    u[0] = np.hstack((ini_h, ini_m))

    rk4(dUdt, 0.01)

    h = u[:, :100]

    # 创建图表对象
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.3)
    plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

    # 初始化曲线
    line, = ax.plot([], [], label="t=")

    x = np.linspace(0.01, 1, 100)


    # 更新函数
    def update(i):
        line.set_data(x, h[i])
        line.set_label("t=" + str(round(i * 0.01, 2)))
        ax.legend()
        return line,


    # 创建动画
    anim = animation.FuncAnimation(fig, update, frames=len(h), interval=100)
    # 保存动画
    mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
                                         codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    anim.save("{}.mp4".format("swe_1d_numerical_02"), writer=mpeg_writer)

    plt.show()

    print()
