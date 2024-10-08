import torch
import numpy as np
import matplotlib.pyplot as plt
from lorenz63_non_chaos_pinn_02 import Lorenz63Net

if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)

    U = np.load('lorenz63_non_chaos.npy')

    t = np.linspace(0, 3, 601)
    u = U[:601]
    x = u[:, 0]
    y = u[:, 1]
    z = u[:, 2]

    model = torch.load('lorenz63_non_chaos_pinn_02_0.0001.pt', map_location=torch.device('cpu'))
    model.eval()
    print("model", model)

    t_test = torch.from_numpy(t[:, None]).float()
    u_hat = model(t_test).detach().numpy()
    x_hat = u_hat[:, 0]
    y_hat = u_hat[:, 1]
    z_hat = u_hat[:, 2]

    epsilon = np.abs(u - u_hat)
    L_inf_error = np.max(epsilon)
    L_2_error = np.sqrt(np.sum(epsilon ** 2) / epsilon.size)
    L_rel_error = np.linalg.norm(epsilon) / np.linalg.norm(u)
    print('L_inf_error :', L_inf_error, ' , L_2_error', L_2_error, ' , L_rel_error : ', L_rel_error)

    # 创建一个 Figure 对象，并设置子图布局
    fig = plt.figure(figsize=(15, 5))
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.plot(t, x, color='b', label='x')
    ax1.plot(t, y, color='g', label='y')
    ax1.plot(t, z, color='k', label='z')
    ax1.plot(t, x_hat, 'r--', label='x_hat')
    ax1.plot(t, y_hat, 'c--', label='y_hat')
    ax1.plot(t, z_hat, color='orange', linestyle='--', label='z_hat')
    ax1.set_xlabel('t', color='black')
    ax1.set_ylabel('u(t)', color='black')
    ax1.legend(loc='upper right')

    ax2.plot(x, y, z, 'r', label='RK')
    ax2.plot(x_hat, y_hat, z_hat, color='b', linestyle='--', label='PINN')
    ax2.legend()
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.set_zlabel('z')

    plt.show()
