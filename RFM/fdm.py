import numpy as np
import math
import time
from numpy.linalg import solve

# analytical solution parameters
AA = 1
aa = 2.0 * np.pi
bb = 3.0 * np.pi

left_boundary = 0.0
right_boundary = 8.0
# 公式中的lambda的平方
lamb = 4


def u(x):
    return AA * np.sin(bb * (x + 0.05)) * np.cos(aa * (x + 0.05)) + 2.0


def d2u_dx2(x):
    return -AA * (aa * aa + bb * bb) * np.sin(bb * (x + 0.05)) * np.cos(aa * (x + 0.05)) \
           - 2.0 * AA * aa * bb * np.cos(bb * (x + 0.05)) * np.sin(aa * (x + 0.05))


def f(x):
    return (d2u_dx2(x) + lamb * u(x))


vanal_u = np.vectorize(u)
vanal_f = np.vectorize(f)


# calculate the matrix A,f in linear equations system 'Au=f'
def cal_matrix(N, points):
    A = np.zeros((N + 1, N + 1), dtype=np.float64)
    dx = (right_boundary - left_boundary) / N
    for i in range(1, N):
        A[i, i - 1] = 1 / (dx ** 2)
        A[i, i] = lamb - 2 / (dx ** 2)
        A[i, i + 1] = 1 / (dx ** 2)
    A[0, 0] = 1.0
    A[N, N] = 1.0
    f = np.zeros((N + 1, 1), dtype=np.float64)
    f[0, 0] = u(points[0])
    f[1:N] = vanal_f(points[1:N]).reshape((-1, 1))
    f[N] = u(points[-1])
    return (A, f)


# calculate the l^{inf}-norm and l^{2}-norm error for u
def test(points, u):
    true_values = vanal_u(points)
    numerical_values = u
    epsilon = true_values - numerical_values
    epsilon = np.maximum(epsilon, -epsilon)
    error_inf = epsilon.max()
    error_l2 = math.sqrt(8 * sum(epsilon * epsilon) / len(epsilon))
    print('L_infty=', error_inf, 'L_2=', error_l2)
    return (error_l2)


def error_plot(multi_Errors):
    import matplotlib.pyplot as plt
    plt.figure(figsize=[7, 5])
    plt.tick_params(labelsize=10)
    font2 = {
        'weight': 'normal',
        'size': 22,
    }
    plt.xlabel('Degrees of freedom', font2)
    plt.ylabel('$L_2$ absolute error', font2)
    plt.xscale('log')
    plt.yscale('log')
    Label = ['FDM', 'PINN', 'RFM']
    for i in range(len(multi_Errors)):
        Error = multi_Errors[i]
        plt.plot(Error[:, 0], Error[:, 1], \
                 lw=1.5, ls='-', clip_on=False, \
                 marker='o', markersize=10, \
                 label=Label[i], \
                 markerfacecolor='none', \
                 markeredgewidth=1.5)
    plt.legend()
    plt.title("Comparison of accuracy on 1D Helmholtz equation")
    plt.show()


def time_plot(multi_Errors):
    import matplotlib.pyplot as plt
    plt.figure(figsize=[7, 5])
    plt.tick_params(labelsize=10)
    font2 = {
        'weight': 'normal',
        'size': 22,
    }
    plt.xlabel('Degrees of freedom', font2)
    plt.ylabel('Solving time', font2)
    Label = ['FDM', 'PINN', 'RFM']
    for i in range(len(multi_Errors)):
        Error = multi_Errors[i]
        plt.plot(Error[:, 0], Error[:, 2], \
                 lw=1.5, ls='-', clip_on=False, \
                 marker='o', markersize=10, \
                 label=Label[i], \
                 markerfacecolor='none', \
                 markeredgewidth=1.5)
    plt.legend()
    plt.title("Comparison of efficiency on 1D Helmholtz equation")
    plt.show()


def main(N):
    time_begin = time.time()
    points = np.linspace(0, 8.0, N + 1)
    A, f = cal_matrix(N, points)
    u = solve(A, f).reshape((-1))
    error = test(points, u)
    time_end = time.time()
    return (error, time_end - time_begin)


if __name__ == '__main__':
    FDM_Error = np.zeros([5, 3])
    for i in range(5):
        N = int(100 * 2 ** i)
        print('***********************')
        print("N = M =", N)
        FDM_Error[i, 0] = N
        FDM_Error[i, 1], FDM_Error[i, 2] = main(N)

    error_plot([FDM_Error])
