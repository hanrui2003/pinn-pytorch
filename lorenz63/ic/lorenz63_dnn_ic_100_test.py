import numpy as np
import torch
import matplotlib.pyplot as plt
from lorenz63_dnn_ic_100 import FCN, SinActivation

if "__main__" == __name__:
    # torch.manual_seed(1234)
    # np.random.seed(1234)

    t_test = torch.linspace(1., 3., 601)

    model = torch.load('lorenz63_dnn_ic_100_0.01.pt', map_location=torch.device('cpu'))
    nn_predict = model(t_test[:, None]).detach().numpy()
    x_nn = nn_predict[:, [0]]
    y_nn = nn_predict[:, [1]]
    z_nn = nn_predict[:, [2]]

    fig, ax = plt.subplots()
    ax.plot(t_test, x_nn, color='r', label='x')
    ax.plot(t_test, y_nn, color='g', label='y')
    ax.plot(t_test, z_nn, color='b', label='z')
    ax.set_title('Lorenz63')
    ax.set_xlabel('t', color='black')
    ax.set_ylabel('f(t)', color='black', rotation=0)
    ax.legend(loc='upper right')
    plt.savefig('./images/lorenz63_dnn_ic_100_2d.png')
    plt.close()
    # plt.show()

    ax = plt.axes(projection="3d")
    ax.plot(x_nn, y_nn, z_nn, color='r')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.set_title('lorenz63')
    plt.savefig('./images/lorenz63_dnn_ic_100_3d.png')
    plt.show()
