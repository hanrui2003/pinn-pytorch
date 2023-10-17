import numpy as np
import matplotlib.pyplot as plt


def RBF(x1, x2, output_scale=1.0, length_scale=0.1):
    """
    高斯径向基函数，用于生成协方差矩阵，l=0.1的时候，一般2-3个波峰和波谷，符合一般的复杂性要求。
    :param x1:随机向量
    :param x2:随机向量
    """
    diffs = np.expand_dims(x1 / length_scale, 1) - np.expand_dims(x2 / length_scale, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


def smooth(x):
    return ((0 <= x) & (x < 1 / 8)) * np.sin(4 * np.pi * x) \
           + ((1 / 8 <= x) & (x < 7 / 8)) * 1 \
           + ((7 / 8 <= x) & (x <= 1)) * np.sin(4 * np.pi * (x - 3 / 4))


if "__main__" == __name__:
    x_lb = 0.
    x_ub = 1.
    # 随机采样的向量维度，也是插值函数的点数
    N = 500
    # 扰动项，防止后续的 cholesky 分解过程中，有些元素非常小甚至为零时，导致平方根运算不稳定。
    jitter = 1e-10
    x = np.linspace(x_lb, x_ub, N)[:, None]
    # 经过核函数运算得到的协方差矩阵
    K = RBF(x, x)
    # cholesky 分解，得到下三角矩阵
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    # 生成N维标准正态分布的向量，与L内积，得到符合协方差为K的正态分布向量
    y_sample = np.dot(L, np.random.randn(N))
    # 把两端的值修改为0，这主要是因为边值为0，这就要求(0,0)和(1,0)的值为0，才能保持连续
    y_sample[[0, -1]] = 0.

    fig, ax1 = plt.subplots()
    ax1.plot(x, y_sample * smooth(x.squeeze(1)), color='blue')
    ax1.plot(x, smooth(x.squeeze(1)), color='red', label='smooth')
    plt.show()
