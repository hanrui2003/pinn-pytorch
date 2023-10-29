import numpy as np
import matplotlib.pyplot as plt


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


def wave_equation():
    alpha = 1
    x0 = 0
    x1 = 6 * np.pi
    l = x1 - x0
    T = 10
    x = np.linspace(x0, x1, 1800)
    t = np.linspace(0, T, 1000)
    X, T = np.meshgrid(x, t)
    U = np.cos(alpha * np.pi * T / l) * np.sin(np.pi * X / l) + (
            np.cos(2 * alpha * np.pi * T / l) + np.sin(2 * alpha * np.pi * T / l) * l / (
            2 * alpha * np.pi)) * np.sin(2 * np.pi * X / l)
    plot(X, T, U)


def heat_equation():
    alpha = np.pi / 2
    x0 = 0
    x1 = 12
    T = 10
    x = np.linspace(x0, x1, 1200)
    t = np.linspace(0, T, 1000)
    X, T = np.meshgrid(x, t)
    U = 2 * np.sin(alpha * X) * np.exp(-T)
    plot(X, T, U)


def convection_diffusion_equation():
    v = 0.01
    D = 0.01
    x0 = 0
    x1 = 1
    T = 1
    x = np.linspace(x0, x1, 100)
    t = np.linspace(0, T, 100)
    X, T = np.meshgrid(x, t)
    U = np.exp(-np.pi ** 2 * D * T) * np.sin(np.pi * X)
    plot(X, T, U)


if __name__ == '__main__':
    # heat_equation()
    # wave_equation()
    convection_diffusion_equation()
