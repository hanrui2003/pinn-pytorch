import csv

import numpy as np
import matplotlib.pyplot as plt  # 2d plot
from mpl_toolkits.mplot3d import Axes3D  # 3d plot

'''
https://github.com/rkikurin/lorenz63/blob/master/lorenz63.py
'''

# define setting
h = 0.01
step = 10000

# define parameters
beta = 8.0 / 3.0
rho = 28.0  # case1
# r=14.0 #case2
# r=1.0 #case3
sigma = 10.0

# define matrix
x = np.empty(step + 1)
y = np.empty(step + 1)
z = np.empty(step + 1)
data = np.empty(3)

f = open('data.csV', 'w')
writer = csv.writer(f, lineterminator='\n')

# define initial
x[0] = 5.0
y[0] = -5.0
z[0] = 25.0

# Forward time step
# Euler method
for i in range(step):
    x[i + 1] = x[i] - sigma * (x[i] - y[i]) * h
    y[i + 1] = y[i] + (-y[i] - x[i] * z[i] + rho * x[i]) * h
    z[i + 1] = z[i] + (x[i] * y[i] - beta * z[i]) * h
    data[0] = x[i + 1]  # for output xt
    data[1] = y[i + 1]  # for output yt
    data[2] = z[i + 1]  # for output zt
    writer.writerow(data)

f.close()

# 2dplot
plt.figure(figsize=(6, 3), dpi=200)
plt.subplot(131)
plt.xlabel('x')  # x name
plt.ylabel('y')  # y name
plt.xlim(-30.0, 30.0)  # x limit
plt.ylim(-30.0, 30.0)  # y limit
plt.plot(x, y, color="r", linewidth=0.5)

plt.subplot(132)
plt.xlabel('x')  # x name
plt.ylabel('z')  # y name
plt.xlim(-30.0, 30.0)  # x limit
plt.ylim(0.0, 60.0)  # y limit
plt.plot(x, z, color="g", linewidth=0.5)

plt.subplot(133)
plt.xlabel('y')  # x name
plt.ylabel('z')  # y name
plt.xlim(-30.0, 30.0)  # x limit
plt.ylim(0.0, 60.0)  # y limit
plt.plot(y, z, color="b", linewidth=0.5)
plt.savefig("output1.png", dpi=200)

plt.show()

# # 3d plot
# fig = plt.figure(figsize=(12, 12), dpi=200)
# ax = Axes3D(fig)
# ax.set_xlabel("x")
# ax.set_ylabel("y")
# ax.set_zlabel("z")
# ax.set_xlim(-30.0, 30.0)  # x limit
# ax.set_ylim(-30.0, 30.0)  # y limit
# ax.set_zlim(0.0, 60.0)  # z limit
# ax.plot3D(x, y, z, color="black", linewidth=0.5)
# plt.savefig("output2.png", dpi=200)
# plt.show()
