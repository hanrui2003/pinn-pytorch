import numpy as np

N_interp = 5001
# 扰动项，防止后续的 cholesky 分解过程中，有些元素非常小甚至为零时，导致平方根运算不稳定。
jitter = 1e-10
x_interp = np.linspace(0, 5, N_interp)
l = 0.2
K = np.exp(-0.5 * (x_interp[:, None] - x_interp) ** 2 / (l ** 2))
# cholesky 分解，得到下三角矩阵
L = np.linalg.cholesky(K + jitter * np.eye(N_interp))
y_interp = np.dot(L, np.random.randn(N_interp))
y_max = y_interp.max()
y_min = y_interp.min()
# 得到值域在[-0.1,0.1]的噪声函数
y_interp = 0.2 * (y_interp - y_min) / (y_max - y_min) - 0.1
np.savez('convection_diffusion_interp.npz', x_interp=x_interp, y_interp=y_interp)
print()
