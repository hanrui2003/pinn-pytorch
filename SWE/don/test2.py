import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 2 * np.pi, 100)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y)  # 绘制曲线或散点图
ax.set_ylim(-1, 3)  # 设置x轴范围为0到10

plt.show()
