import numpy as np


# 定义随机函数
def random_function(x):
    return np.random.rand(*x.shape) * np.random.choice([-1, 1], size=x.shape)


# 定义边界条件
def boundary_condition(x):
    return np.piecewise(x, [x < 0, x == 0, x > 0], [0, 0, 0])


# 生成随机函数
x = np.linspace(-1, 1, 1000)
y = random_function(x)

# 应用边界条件
y = y - boundary_condition(x)

# 绘制随机函数
import matplotlib.pyplot as plt

plt.plot(x, y)
plt.show()
