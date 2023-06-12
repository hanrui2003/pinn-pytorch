import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 101)
y = -np.sin(x * np.pi * 2) + 2

fig, ax = plt.subplots()
ax.set_xlim(0, 1)
ax.set_ylim(0, 3)
plt.grid(True, color='gray', linestyle='--', linewidth=0.5)

# 初始化曲线
ax.plot(x, y, label="t=")
plt.show()
