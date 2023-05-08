import numpy as np
from sklearn.neighbors import KernelDensity

# 生成一些随机向量的样本
X = np.random.randn(1000, 2)

# 拟合KDE模型
kde = KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)

# 生成一些测试点
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
xy = np.column_stack([xx.ravel(), yy.ravel()])

# 使用KDE模型来估计每个测试点的概率密度
Z = np.exp(kde.score_samples(xy))
Z = Z.reshape(xx.shape)

# 可视化概率密度估计
import matplotlib.pyplot as plt
plt.contourf(xx, yy, Z, cmap='viridis')
plt.colorbar()
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.show()
