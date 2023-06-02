import numpy as np
import matplotlib.pyplot as plt

# mesh parameter
L = 1
div = 100
dx = L / div

# time parameter
t0 = 0.0
T = 5

# physics constants
g = 1.0


# initial height profile
def u0(x):
    return 0.1 + 0.1 * np.exp(-64 * (x - 0.25) * (x - 0.25))


# initial height vector
ini_h = [u0(i * dx - 0.5 * dx) for i in range(1, div + 1)]
# initial velocity vector
ini_m = [0 for i in range(div)]

ini_U = np.concatenate((ini_h, ini_m))


def f(H, M, i):
    return M[i], M[i] * M[i] / H[i] + (g / 2) * H[i] * H[i]


def flux(H, M, L, R):
    if L == 0:
        return 0, f(H, M, R)[1]
    if R == div + 1:
        return 0, f(H, M, L)[1]
    fl = f(H, M, L)
    fr = f(H, M, R)
    d = 0.2
    return (fl[0] + fr[0]) / 2 + d * (H[L] - H[R]) / 2, (fl[1] + fr[1]) / 2 + d * (M[L] - M[R]) / 2


def dUdt(U, t):
    curr_dhdt = []
    curr_dmdt = []
    curr_h = U[:div]
    curr_m = U[div:2 * div]
    for i in range(div):
        flux_l = flux(curr_h, curr_m, i - 1, i)
        flux_r = flux(curr_h, curr_m, i, i + 1)
        curr_dhdt.append((flux_l[0] - flux_r[0]) / dx)
        curr_dmdt.append((flux_l[1] - flux_r[1]) / dx)
    return np.concatenate((curr_dhdt, curr_dmdt))


def rk4(f, t0, t1, y0, h):
    t = [t0]
    y = np.reshape(y0, (1, len(y0)))
    for tm in np.arange(t0 + h, t1 + h, h):
        k1 = f(y[-1, :], tm)
        k2 = f(y[-1, :] + 0.5 * h * k1, tm + 0.5 * h)
        k3 = f(y[-1, :] + 0.5 * h * k2, tm + 0.5 * h)
        k4 = f(y[-1, :] + h * k3, tm + h)
        y = np.concatenate((y, [np.transpose(y[-1, :] + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6)]))
        t.append(tm)
    return t, y


t, y = rk4(dUdt, t0, T, ini_U, 0.01)

for i in range(1, 501):
    plt.plot([dx * i for i in range(1, div + 1)], y[i, :div], label="t=" + str(round(i * 0.01, 2)))
    plt.ylim(0, 0.3)
    plt.legend()
    plt.show()

    print("=============")
