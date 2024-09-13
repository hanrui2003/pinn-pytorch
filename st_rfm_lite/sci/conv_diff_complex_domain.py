import numpy as np
import matplotlib.pyplot as plt

# 定义矩形区域
x_min, x_max = 0, 4
y_min, y_max = 0, 4

# 定义三个圆的参数（圆心和半径）
circles = [
    {'center': (1, 1), 'radius': 0.5},  # 完全包含在矩形内
    {'center': (3, 2), 'radius': 0.8},  # 完全包含在矩形内
    {'center': (2, 3.5), 'radius': 0.7}  # 部分包含在矩形内
]

# 生成矩形区域内的网格点
x_obs = np.linspace(x_min, x_max, 9)[1:-1]
y_obs = np.linspace(y_min, y_max, 9)[1:-1]
X_obs, Y_obs = np.meshgrid(x_obs, y_obs)

# 初始化一个布尔数组，用于标记在圆外的点
outside = np.ones_like(X_obs, dtype=bool)

# 对每个圆，标记出在圆内的点
for circle in circles:
    x0, y0 = circle['center']
    r = circle['radius']
    distances = np.sqrt((X_obs - x0) ** 2 + (Y_obs - y0) ** 2)
    outside = outside & (distances >= r)

# 筛选出位于所有圆外的点
X_obs_outside = X_obs[outside]
Y_obs_outside = Y_obs[outside]

print("number of obs :", len(X_obs_outside))

# 图1：显示取的点
plt.figure(figsize=(6, 6))
plt.plot(X_obs_outside, Y_obs_outside, 'b.', markersize=5)

# 画出每个圆
for circle in circles:
    x0, y0 = circle['center']
    r = circle['radius']
    circ = plt.Circle((x0, y0), r, color='r', fill=False, linewidth=1)
    plt.gca().add_patch(circ)

# 设置图形参数
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')
# plt.axis('off')
plt.show()

x = np.linspace(x_min, x_max, 500)
y = np.linspace(y_min, y_max, 500)
X, Y = np.meshgrid(x, y)

# 初始化一个布尔数组，用于标记在圆外的点
outside = np.ones_like(X, dtype=bool)

# 对每个圆，标记出在圆内的点
for circle in circles:
    x0, y0 = circle['center']
    r = circle['radius']
    distances = np.sqrt((X - x0) ** 2 + (Y - y0) ** 2)
    outside = outside & (distances >= r)

# 图2：剩余区域用纯蓝色背景表示
plt.figure(figsize=(6, 6))

# 使用contourf绘制背景
plt.contourf(X, Y, outside, levels=[0, 0.5, 1], colors=['white', 'blue'])

# 画出每个圆
for circle in circles:
    x0, y0 = circle['center']
    r = circle['radius']
    circ = plt.Circle((x0, y0), r, color='r', fill=False, linewidth=1)
    plt.gca().add_patch(circ)

# 设置图形参数
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('x')
plt.ylabel('y')
plt.gca().set_aspect('equal', adjustable='box')
# plt.axis('off')
plt.show()
