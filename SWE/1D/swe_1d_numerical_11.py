import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib import animation

N = 100
Nt = 500
T_max = 0.5

g = 100
N = N
Nt = Nt
T_max = T_max
x = np.linspace(0, 1, N + 1, endpoint=True)
dx = x[-1] - x[-2]
t = np.linspace(0, T_max, Nt + 1)
xc = (x[:-1] + x[1:]) / 2
# u0 = np.concatenate([-np.sin(xc * np.pi * 2) + 2, np.zeros_like(-np.sin(xc * np.pi * 2) + 2)])
h0 = 0.1 + 0.1 * np.exp(-64 * (xc - 0.25) ** 2)
m0 = np.zeros_like(h0)
u0 = np.concatenate([h0, m0])


def dudt_of_shallow_water(t, w):
    n = len(w) // 2
    h = w[:n]
    u = w[n:] / h
    F1 = h * u
    F2 = h * u * u + h ** 2 / 2 * g
    J = np.abs(u) + np.sqrt(h * g)
    J_interface = np.maximum(J[:-1], J[1:])
    w1 = h
    F1_interface = (F1[:-1] + F1[1:]) / 2 + J_interface * (w1[:-1] - w1[1:]) / 2
    w2 = h * u
    F2_interface = (F2[:-1] + F2[1:]) / 2 + J_interface * (w2[:-1] - w2[1:]) / 2
    F1_interface = np.concatenate([[0], F1_interface, [0]])
    F2_interface = np.concatenate([[h[0] ** 2 / 2 * g], F2_interface, [h[-1] ** 2 / 2 * g]])
    return np.concatenate([F1_interface[:-1] - F1_interface[1:],
                           F2_interface[:-1] - F2_interface[1:]]) / dx


res = solve_ivp(dudt_of_shallow_water, t_span=(0, T_max), y0=u0, t_eval=t)
u = res['y'].T

h = u[:, :100]

# 创建图表对象
fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, .3)
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
anim = animation.FuncAnimation(fig, update, frames=len(h), interval=20)
# 保存动画
mpeg_writer = animation.FFMpegWriter(fps=24, bitrate=10000,
                                     codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
anim.save("{}.mp4".format("swe_1d_numerical_11"), writer=mpeg_writer)

plt.show()
