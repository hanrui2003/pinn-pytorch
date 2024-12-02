import matplotlib.pyplot as plt
# 使用 macOS 中的中文字体
# plt.rcParams['font.family'] = 'AppleGothic'

# 准备数据
x = [50, 100, 150, 200, 250]
y1 = [6.6e-2, 5.7e-4, 8.89e-6, 3.46e-7, 4.21e-8]
y2 = [7.82e-2, 4.83e-4, 1.29e-5, 1.54e-6, 2.14e-7]
y3 = [2.37e-4, 1.26e-7, 3.51e-9, 9.48e-10, 3.59e-10]

# 绘制三条折线，分别使用不同颜色和菱形标记
plt.plot(x, y1, marker='D', color='blue', label=r'PoU-1')
plt.plot(x, y2, marker='D', color='green', label=r'PoU-2')
plt.plot(x, y3, marker='D', color='red', label=r'IRFM')

# 设置横坐标为对数刻度
plt.yscale('log', base=10)

# 添加标签和标题
plt.xlabel('Number of feature functions')
plt.ylabel(r'$L^\infty$ error')
# plt.title('Multiple Lines with Diamond Markers and Different Colors (Logarithmic X Axis)')

# 显示图例
plt.legend()

plt.show()

y1 = [1.64e-2, 7.17e-5, 9.73e-7, 3.93e-8, 6.46e-9]
y2 = [1.58e-2, 5.87e-5, 1.77e-6, 2.03e-7, 2.97e-8]
y3 = [2.18e-5, 1.26e-8, 2.78e-10, 8.54e-11, 6.41e-11]

# 绘制三条折线，分别使用不同颜色和菱形标记
plt.plot(x, y1, marker='D', color='blue', label=r'PoU-1')
plt.plot(x, y2, marker='D', color='green', label=r'PoU-2')
plt.plot(x, y3, marker='D', color='red', label=r'IRFM')

# 设置横坐标为对数刻度
plt.yscale('log', base=10)

# 添加标签和标题
plt.xlabel('Number of feature functions')
plt.ylabel(r'$L^2$ error')

# 显示图例
plt.legend()

# 显示图表
plt.show()
