"""
用于EAJAM
"""
import matplotlib.pyplot as plt
# 使用 macOS 中的中文字体
# plt.rcParams['font.family'] = 'AppleGothic'

# 准备数据
x = [2, 4, 8, 16, 32]
y1 = [39.58, 1.20e-2, 1.77e-5, 1.66e-8, 2.15e-9]
y2 = [8, 1.26e-4, 2.14e-7, 1.38e-9, 1.27e-10]
y3 = [0.34, 5.48e-8, 6.43e-10, 3.17e-13, 3.89e-14]

# 绘制三条折线，分别使用不同颜色和菱形标记
plt.plot(x, y1, marker='D', color='blue', label=r'PoU-1')
plt.plot(x, y2, marker='D', color='green', label=r'PoU-2')
plt.plot(x, y3, marker='D', color='red', label=r'IRFM')

# 设置横坐标为对数刻度
plt.xscale('log', base=2)
plt.yscale('log', base=10)

# 添加标签和标题
plt.xlabel('Number of partition')
plt.ylabel(r'$L^\infty$ error')
# plt.title('Multiple Lines with Diamond Markers and Different Colors (Logarithmic X Axis)')

# 显示图例
plt.legend()

plt.show()

y1 = [10.55, 3.09e-3, 4.5e-6, 4.75e-9, 8.69e-10]
y2 = [1.58, 1.51e-5, 3.68e-8, 2.22e-10, 2.64e-11]
y3 = [0.057, 1.11e-8, 1.04e-10, 1.46e-13, 1.31e-14]

# 绘制三条折线，分别使用不同颜色和菱形标记
plt.plot(x, y1, marker='D', color='blue', label=r'PoU-1')
plt.plot(x, y2, marker='D', color='green', label=r'PoU-2')
plt.plot(x, y3, marker='D', color='red', label=r'IRFM')

# 设置横坐标为对数刻度
plt.xscale('log', base=2)
plt.yscale('log', base=10)

# 添加标签和标题
plt.xlabel('Number of partition')
plt.ylabel(r'$L^2$ error')

# 显示图例
plt.legend()

# 显示图表
plt.show()
