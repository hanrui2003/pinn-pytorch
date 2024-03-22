import numpy as np
import matplotlib.pyplot as plt


# 定义函数
def psi_a(x):
    return np.where(np.logical_and(x >= -1, x <= 1), 1, 0)


def psi_b(x):
    return np.where(x < -5 / 4, 0,
                    np.where(np.logical_and(x >= -5 / 4, x < -3 / 4), (1 + np.sin(2 * np.pi * x)) / 2,
                             np.where(np.logical_and(x >= -3 / 4, x < 3 / 4), 1,
                                      np.where(np.logical_and(x >= 3 / 4, x < 5 / 4), (1 - np.sin(2 * np.pi * x)) / 2,
                                               0))))


# 生成x值
x = np.linspace(-1.5, 1.5, 1000)

# 绘图
fig = plt.figure(figsize=(10, 3))
plt.plot(x, psi_a(x), label=r'$\psi_{n}^{a}$', linestyle='-')
plt.plot(x, psi_b(x), label=r'$\psi_{n}^{b}$', linestyle='--')
plt.xlabel('x')
plt.ylabel(r'$\psi$')
# plt.title('Functions')
plt.legend()
plt.grid(True)
plt.show()
