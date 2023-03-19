import matplotlib.pyplot as plt
import numpy as np
# 绘制三维图像
import mpl_toolkits.mplot3d as p3d

'''
https://blog.csdn.net/qq_41137110/article/details/115392340
Lorenz吸引子生成函数
参数为三个初始坐标，三个初始参数,迭代次数
返回三个一维list
'''


def Lorenz(x0, y0, z0, p, q, r, T):
    # 微分迭代步长
    h = 0.01
    x = []
    y = []
    z = []
    for t in range(T):
        xt = x0 + h * p * (y0 - x0)
        yt = y0 + h * (q * x0 - y0 - x0 * z0)
        zt = z0 + h * (x0 * y0 - r * z0)
        # x0、y0、z0统一更新
        x0, y0, z0 = xt, yt, zt
        x.append(x0)
        y.append(y0)
        z.append(z0)
    return x, y, z


def main():
    # 设定参数
    p = 10
    q = 28
    r = 8 / 3
    # 迭代次数
    T = 10000
    # 设初值
    x0 = -16
    y0 = -21
    z0 = 33
    # fig=plt.figure()
    # ax=p3d.Axes3D(fig)
    x, y, z = Lorenz(x0, y0, z0, p, q, r, T)
    ax = plt.subplot(121, projection="3d")
    ax.scatter(x, y, z, s=5)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_zlabel('z(t)')
    ax.set_title('x0=-16 y0=-21 z0=33')
    # plt.axis('off')
    # 消除网格

    ax.grid(False)
    # 初值微小的变化
    x0 = -16
    y0 = -21
    z0 = 33.00001
    xx, yy, zz = Lorenz(x0, y0, z0, p, q, r, T)
    ax = plt.subplot(122, projection="3d")
    ax.scatter(xx, yy, zz, s=5)
    ax.set_xlabel('x(t)')
    ax.set_ylabel('y(t)')
    ax.set_zlabel('z(t)')
    ax.set_title('x0=-16 y0=-21 z0=33.00001')
    ax.grid(False)
    plt.show()

    t = np.arange(0, T)
    plt.scatter(t, x, s=1)
    plt.scatter(t, xx, s=1)
    plt.show()


if __name__ == '__main__':
    main()
