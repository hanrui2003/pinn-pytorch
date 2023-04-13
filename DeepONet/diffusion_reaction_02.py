"""
主要用于代码分析，增加注释
"""
import jax
import jax.numpy as np
from jax import random, grad, vmap, jit, hessian, lax
from jax.example_libraries import optimizers
from jax.nn import relu
from jax.config import config
from jax.numpy import index_exp as index
from jax.flatten_util import ravel_pytree
import itertools
from functools import partial
from torch.utils import data
from tqdm import trange, tqdm
import matplotlib.pyplot as plt

from scipy.interpolate import griddata

from jax.lax import stop_gradient


def plot(X, T, f):
    fig = plt.figure(figsize=(7, 5))
    plt.pcolor(X, T, f, cmap='rainbow')
    plt.colorbar()
    plt.xlabel('$x$')
    plt.ylabel('$t$')
    plt.title('$f(x,t)$')
    plt.tight_layout()


# Use double precision to generate data (due to GP sampling)
# 高斯过程的核函数-径向基函数（radial basis function, RBF）
# 核函数就是协方差函数，任意两个时刻对应的高斯随机变量的协方差矩阵
# 使用径向基，得到的是正定，自己和自己做协方差，不一定得到单位阵
# output_scale 控制函数的范围
def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - \
            np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * np.exp(-0.5 * r2)


# A diffusion-reaction numerical solver
# 使用零均值高斯过程构造同分布随机变量函数，然后求微分方程的解
def solve_ADR(key, Nx, Nt, P, length_scale):
    """Solve 1D
    u_t = (k(x) u_x)_x - (v(x) u)_x + g(u) + f(x)
    with zero initial and boundary conditions.
    """
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    # 函数：返回与输入同形的向量（或标量），每个元素为0.01
    k = lambda x: 0.01 * np.ones_like(x)
    # 零函数：返回与输入同形的零向量（或标量）
    v = lambda x: np.zeros_like(x)
    # 函数g(u)=0.01*u^2
    g = lambda u: 0.01 * u ** 2
    # 导函数g'(u) = 0.02*u
    dg = lambda u: 0.02 * u
    # 零函数：返回与输入同形的零向量（或标量）
    u0 = lambda x: np.zeros_like(x)
    # 测试
    # u0 = lambda x: np.ones_like(x)

    # Generate subkeys
    # JAX的key不能重用，不然每次生成的都一样，所以每次要用随机数的时候，都要生成新的子key
    # 以下代码是分成两个新的key，用来生成随机数
    subkeys = random.split(key, 2)

    # Generate a GP sample
    # 用于生成随机函数，做插值函数的点数
    N = 512
    gp_params = (1.0, length_scale)
    # 增加一点小的扰动
    jitter = 1e-10
    X = np.linspace(xmin, xmax, N)[:, None]
    K = RBF(X, X, gp_params)
    L = np.linalg.cholesky(K + jitter * np.eye(N))
    # 这里是生成一个N维向量满足多元正态分布N(0,K)
    gp_sample = np.dot(L, random.normal(subkeys[0], (N,)))
    # Create a callable interpolation function
    # 一维分段线性差值
    # 和numpy的interp差不多，第二第三个参数是用来做差值函数的横纵坐标，第一个参数是要进行估计的x值
    # 纵坐标跟横坐标同分布
    f_fn = lambda x: np.interp(x, X.flatten(), gp_sample)

    # Create grid
    x = np.linspace(xmin, xmax, Nx)
    t = np.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    # Compute coefficients and forcing
    # 函数：返回与输入同形的向量（或标量），每个元素为0.01
    k = k(x)
    # 零函数：返回与输入同形的零向量（或标量）
    v = v(x)
    f = f_fn(x)

    # Compute finite difference operators
    D1 = np.eye(Nx, k=1) - np.eye(Nx, k=-1)
    D2 = -2 * np.eye(Nx) + np.eye(Nx, k=-1) + np.eye(Nx, k=1)
    D3 = np.eye(Nx - 2)
    M = -np.diag(D1 @ k) @ D1 - 4 * np.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v_bond = 2 * h * np.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * np.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

    # Initialize solution and apply initial condition
    u = np.zeros((Nx, Nt))
    u = u.at[:, 0].set(u0(x))

    # u = index_update(u, index[:,0], u0(x))
    # Time-stepping update
    def body_fn(i, u):
        # 函数g(u)=0.01*u^2
        gi = g(u[1:-1, i])
        # 导函数g'(u) = 0.02*u
        dgi = dg(u[1:-1, i])
        h2dgi = np.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u = u.at[1:-1, i + 1].set(np.linalg.solve(A, b1 + b2))
        # u = index_update(u, index[1:-1, i + 1], np.linalg.solve(A, b1 + b2))
        return u

    # Run loop
    # UU的横向是t，纵向是x，坐标原点应该是左上角
    UU = lax.fori_loop(0, Nt - 1, body_fn, u)

    # for i in range(Nt):
    #     gi = g(u[1:-1, i])
    #     # 导函数g'(u) = 0.02*u
    #     dgi = dg(u[1:-1, i])
    #     h2dgi = np.diag(4 * h2 * dgi)
    #     A = mv_bond - h2dgi
    #     b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
    #     b2 = (c - h2dgi) @ u[1:-1, i].T
    #     u = u.at[1:-1, i + 1].set(np.linalg.solve(A, b1 + b2))

    # Input sensor locations and measurements
    # 输入的源项函数，对应的其实本质就是方程中的f(x)，这里变量的名字有点滥用，注意区分
    xx = np.linspace(xmin, xmax, m)
    u = f_fn(xx)
    # Output sensor locations and measurements
    idx = random.randint(subkeys[1], (P, 2), 0, max(Nx, Nt))
    # 在PDE解的定义域内随机取点(x,t)，这里取的电不一定是边界上的，但是这些都是有label的点
    # 这边和讲义不符，讲义上说的是初边值条件点
    y = np.concatenate([x[idx[:, 0]][:, None], t[idx[:, 1]][:, None]], axis=1)
    # y对应的函数值
    s = UU[idx[:, 0], idx[:, 1]]
    # x, t: sampled points on grid
    # x,t对应网格的坐标，x总坐标，t横坐标，UU对应的PDE数值解的函数值
    return (x, t, UU), (u, y, s)


# Geneate training data corresponding to one input sample
def generate_one_training_data(key, P, Q):
    # Numerical solution
    (x, t, UU), (u, y, s) = solve_ADR(key, Nx, Nt, P, length_scale)

    # Geneate subkeys
    subkeys = random.split(key, 4)

    # Sample points from the boundary and the inital conditions
    # Here we regard the initial condition as a special type of boundary conditions
    x_bc1 = np.zeros((P // 3, 1))
    x_bc2 = np.ones((P // 3, 1))
    x_bc3 = random.uniform(key=subkeys[0], shape=(P // 3, 1))
    x_bcs = np.vstack((x_bc1, x_bc2, x_bc3))

    t_bc1 = random.uniform(key=subkeys[1], shape=(P // 3 * 2, 1))
    t_bc2 = np.zeros((P // 3, 1))
    t_bcs = np.vstack([t_bc1, t_bc2])

    # Training data for BC and IC
    u_train = np.tile(u, (P, 1))
    y_train = np.hstack([x_bcs, t_bcs])
    s_train = np.zeros((P, 1))

    # Sample collocation points
    x_r_idx = random.choice(subkeys[2], np.arange(Nx), shape=(Q, 1))
    x_r = x[x_r_idx]
    t_r = random.uniform(subkeys[3], minval=0, maxval=1, shape=(Q, 1))

    # Training data for the PDE residual
    '''For the operator'''
    u_r_train = np.tile(u, (Q, 1))
    y_r_train = np.hstack([x_r, t_r])
    '''For the function'''
    f_r_train = u[x_r_idx]
    return u_train, y_train, s_train, u_r_train, y_r_train, f_r_train


# Geneate test data corresponding to one input sample
def generate_one_test_data(key, P):
    Nx = P
    Nt = P
    (x, t, UU), (u, y, s) = solve_ADR(key, Nx, Nt, P, length_scale)

    XX, TT = np.meshgrid(x, t)

    u_test = np.tile(u, (P ** 2, 1))
    y_test = np.hstack([XX.flatten()[:, None], TT.flatten()[:, None]])
    s_test = UU.T.flatten()

    return u_test, y_test, s_test


# Data generator
class DataGenerator(data.Dataset):
    def __init__(self, u, y, s,
                 batch_size=64, rng_key=random.PRNGKey(1234)):
        'Initialization'
        self.u = u  # input sample
        self.y = y  # location
        self.s = s  # labeled data evulated at y (solution measurements, BC/IC conditions, etc.)

        self.N = u.shape[0]
        self.batch_size = batch_size
        self.key = rng_key

    def __getitem__(self, index):
        'Generate one batch of data'
        self.key, subkey = random.split(self.key)
        inputs, outputs = self.__data_generation(subkey)
        return inputs, outputs

    @partial(jit, static_argnums=(0,))
    def __data_generation(self, key):
        """
        生成器函数，类似于DataLoader
        用于生成包含批次数据的输入和输出。该函数接受一个随机数种子作为参数，
        从给定的数据集中随机选择batch_size个样本，
        并将它们的特征、标签和状态构造成一个输入元组 (u, y) 和一个输出数组 s。
        :param key:
        :return:
        """
        idx = random.choice(key, self.N, (self.batch_size,), replace=False)
        s = self.s[idx, :]
        y = self.y[idx, :]
        u = self.u[idx, :]
        # Construct batch
        inputs = (u, y)
        outputs = s
        return inputs, outputs


# Define the neural net
# 就是简单的DNN
def MLP(layers, activation=relu):
    ''' Vanilla MLP'''

    def init(rng_key):
        def init_layer(key, d_in, d_out):
            k1, k2 = random.split(key)
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            W = glorot_stddev * random.normal(k1, (d_in, d_out))
            b = np.zeros(d_out)
            return W, b

        key, *keys = random.split(rng_key, len(layers))
        params = list(map(init_layer, keys, layers[:-1], layers[1:]))
        return params

    def apply(params, inputs):
        for W, b in params[:-1]:
            outputs = np.dot(inputs, W) + b
            inputs = activation(outputs)
        W, b = params[-1]
        outputs = np.dot(inputs, W) + b
        return outputs

    return init, apply


# Define the model
class PI_DeepONet:
    def __init__(self, branch_layers, trunk_layers):
        # Network initialization and evaluation functions
        self.branch_init, self.branch_apply = MLP(branch_layers, activation=np.tanh)
        self.trunk_init, self.trunk_apply = MLP(trunk_layers, activation=np.tanh)

        # Initialize
        branch_params = self.branch_init(rng_key=random.PRNGKey(1234))
        trunk_params = self.trunk_init(rng_key=random.PRNGKey(4321))
        params = (branch_params, trunk_params)

        # Use optimizers to set optimizer initialization and update functions
        self.opt_init, \
        self.opt_update, \
        self.get_params = optimizers.adam(optimizers.exponential_decay(1e-3,
                                                                       decay_steps=2000,
                                                                       decay_rate=0.9))
        self.opt_state = self.opt_init(params)

        # Used to restore the trained model parameters
        _, self.unravel_params = ravel_pytree(params)

        self.itercount = itertools.count()

        # Loggers
        # 存储数据，用于恢复损失函数
        self.loss_log = []
        self.loss_bcs_log = []
        self.loss_res_log = []

    # Define DeepONet architecture
    # 参数是分支网络和主干网路的输入，u和y，其中y=(x,t)
    # 就是网络的forward
    def operator_net(self, params, u, x, t):
        branch_params, trunk_params = params
        y = np.stack([x, t])
        # 分支网络的forward
        B = self.branch_apply(branch_params, u)
        # 主干网络的forward
        T = self.trunk_apply(trunk_params, y)
        # 算子网络的输出为分支网络和主干网络输出的点积
        outputs = np.sum(B * T)
        return outputs

    # Define ODE/PDE residual
    # loss_physics
    def residual_net(self, params, u, x, t):
        s = self.operator_net(params, u, x, t)
        s_t = grad(self.operator_net, argnums=3)(params, u, x, t)
        s_x = grad(self.operator_net, argnums=2)(params, u, x, t)
        s_xx = grad(grad(self.operator_net, argnums=2), argnums=2)(params, u, x, t)

        res = s_t - 0.01 * s_xx - 0.01 * s ** 2
        return res

    # Define boundary loss
    def loss_bcs(self, params, batch):
        inputs, outputs = batch
        u, y = inputs

        # Compute forward pass
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])

        # Compute loss
        loss = np.mean((outputs.flatten() - s_pred) ** 2)
        return loss

    # Define residual loss
    def loss_res(self, params, batch):
        inputs, outputs = batch
        u, y = inputs
        # Compute forward pass
        pred = vmap(self.residual_net, (None, 0, 0, 0))(params, u, y[:, 0], y[:, 1])

        # Compute loss
        loss = np.mean((outputs.flatten() - pred) ** 2)
        return loss

        # Define total loss

    def loss(self, params, bcs_batch, res_batch):
        loss_bcs = self.loss_bcs(params, bcs_batch)
        loss_res = self.loss_res(params, res_batch)
        loss = loss_bcs + loss_res
        return loss

        # Define a compiled update step

    # 根据损失反向传播，并更新网络参数
    @partial(jit, static_argnums=(0,))
    def step(self, i, opt_state, bcs_batch, res_batch):
        params = self.get_params(opt_state)
        # 函数式编程，对self.loss关于它的某个自变量求导，默认对第0个变量，也就是此处的params求导。
        # 这里的params就是网络的weight和bias
        g = grad(self.loss)(params, bcs_batch, res_batch)
        return self.opt_update(i, g, opt_state)

    # Optimize parameters in a loop
    def train(self, bcs_dataset, res_dataset, nIter=10000):
        # Define data iterators
        bcs_data = iter(bcs_dataset)
        res_data = iter(res_dataset)

        pbar = trange(nIter)
        # Main training loop
        for it in pbar:
            # Fetch data
            bcs_batch = next(bcs_data)
            res_batch = next(res_data)

            self.opt_state = self.step(next(self.itercount), self.opt_state, bcs_batch, res_batch)

            if it % 100 == 0:
                params = self.get_params(self.opt_state)

                # Compute losses
                loss_value = self.loss(params, bcs_batch, res_batch)
                loss_bcs_value = self.loss_bcs(params, bcs_batch)
                loss_res_value = self.loss_res(params, res_batch)

                # Store losses
                self.loss_log.append(loss_value)
                self.loss_bcs_log.append(loss_bcs_value)
                self.loss_res_log.append(loss_res_value)

                # Print losses
                pbar.set_postfix({'Loss': loss_value,
                                  'loss_bcs': loss_bcs_value,
                                  'loss_physics': loss_res_value})

    # Evaluates predictions at test points
    @partial(jit, static_argnums=(0,))
    def predict_s(self, params, U_star, Y_star):
        s_pred = vmap(self.operator_net, (None, 0, 0, 0))(params, U_star, Y_star[:, 0], Y_star[:, 1])
        return s_pred

    @partial(jit, static_argnums=(0,))
    def predict_res(self, params, U_star, Y_star):
        r_pred = vmap(self.residual_net, (None, 0, 0, 0))(params, U_star, Y_star[:, 0], Y_star[:, 1])
        return r_pred


# Resolution of the solution (Grid of 100x100)
Nx = 100
Nt = 100
N = 1  # number of input samples
# Select the number of sensors
m = Nx  # number of input sensors
# 注意这里不一定是初边值点，和讲义中的略有不同
P_train = 300  # number of output sensors, 100 for each side
Q_train = 100  # number of collocation points for each input sample
# GRF length scale
length_scale = 0.2

# jax浮点数默认是32位，这里暂时修改为64位
config.update("jax_enable_x64", True)

# PRNGKey会生成一个（2，）shape array来作为seed的值，在未来需要生成随机数的时候，可以直接使用key值来作为seed，方便操作。
# 下面的key为[0 0]，类似于Numpy的中子数，Numpy是面向Python的，串行的。
# 但是在JAX里面，是并行计算的，所以随机数不能用全局状态，必须显示指定各自的key，就类似numpy的seed
key = random.PRNGKey(0)
keys = random.split(key, N)

# 不同的key得到不同的输入函数u，对应的UU是PDE的数值解
# 展示如何生成一个训练样本，非主逻辑 start
(x, t, UU), (u, y, s) = solve_ADR(key, Nx, Nt, P_train, length_scale)
# XX, TT = np.meshgrid(x, t)
# plot(XX, TT, UU)

# Sample points from the boundary and the inital conditions
# Geneate subkeys
subkeys = random.split(key, 4)

# loss operator start
# Here we regard the initial condition as a special type of boundary conditions
# x=0，与下面的t_bc1 相对
x_bc1 = np.zeros((P_train // 3, 1))
# x=1，与下面的t_bc1 相对
x_bc2 = np.ones((P_train // 3, 1))
# x随机取值
x_bc3 = random.uniform(key=subkeys[0], shape=(P_train // 3, 1))
x_bcs = np.vstack((x_bc1, x_bc2, x_bc3))
# Time
# 取随机取值
t_bc1 = random.uniform(key=subkeys[1], shape=(P_train // 3 * 2, 1))  # 200 points
# t=0，与上面的x_bc3相对
t_bc2 = np.zeros((P_train // 3, 1))  # 100 points
t_bcs = np.vstack([t_bc1, t_bc2])  # 末尾的s只是表示复数

# Training data for BC and IC
u_train = np.tile(u, (P_train, 1))  # Add dimentions-> copy u P times ，m points
# 对应讲义中的y_u，相当于一般PINN里面的x_train
y_train = np.hstack([x_bcs, t_bcs])  # stack the  P output sensors
# 相当于一般PINN里面的y_train
s_train = np.zeros((P_train, 1))  # Remember that the initial conditions are 0

# loss operator end

# loss physics start
# Sample collocation points
# 随机取x和t的方式不同，对于x的取法，是由于后续要拿源项u(x)的值
x_r_idx = random.choice(subkeys[2], np.arange(Nx), shape=(Q_train, 1), replace=False)
x_r = x[x_r_idx]
t_r = random.uniform(subkeys[3], minval=0, maxval=1, shape=(Q_train, 1))

# Training data for the PDE residual
u_r_train = np.tile(u, (Q_train, 1))
y_r_train = np.hstack([x_r, t_r])
f_r_train = u[x_r_idx]

# loss physics end

# 展示如何生成一个训练样本，非主逻辑 end

# vmap 可以重复调用函数多次
u_train, y_train, s_train, u_r_train, y_r_train, f_r_train = vmap(generate_one_training_data, (0, None, None))(keys,
                                                                                                               P_train,
                                                                                                               Q_train)
# Reshape the data
u_bcs_train = np.float32(u_train.reshape(N * P_train, -1))
y_bcs_train = np.float32(y_train.reshape(N * P_train, -1))
s_bcs_train = np.float32(s_train.reshape(N * P_train, -1))

u_res_train = np.float32(u_r_train.reshape(N * Q_train, -1))
y_res_train = np.float32(y_r_train.reshape(N * Q_train, -1))
f_res_train = np.float32(f_r_train.reshape(N * Q_train, -1))

N_test = 1  # number of input samples
key = random.PRNGKey(12345)  # it should be different than the key we used for the training data
P_test = 100
keys = random.split(key, N_test)

u_test, y_test, s_test = vmap(generate_one_test_data, (0, None))(keys, P_test)

# Reshape Data
u_test = np.float32(u_test.reshape(N_test * P_test ** 2, -1))
y_test = np.float32(y_test.reshape(N_test * P_test ** 2, -1))
s_test = np.float32(s_test.reshape(N_test * P_test ** 2, -1))

config.update("jax_enable_x64", False)

# Initialize model
branch_layers = [m, 5, 5]
trunk_layers = [2, 4, 4]
model = PI_DeepONet(branch_layers, trunk_layers)

# Create data set
batch_size = 10000
bcs_dataset = DataGenerator(u_bcs_train, y_bcs_train, s_bcs_train, batch_size)
res_dataset = DataGenerator(u_res_train, y_res_train, f_res_train, batch_size)

# Train
model.train(bcs_dataset, res_dataset, nIter=40000)

params = model.get_params(model.opt_state)
s_pred = model.predict_s(params, u_test, y_test)[:, None]
error_s = np.linalg.norm(s_test - s_pred) / np.linalg.norm(s_test)
print(error_s)

N_test = 1  # number of input samples
# 这边的key参数可以用于改变输入函数，即不同的输入函数经过算子运算得到不用的PDE的解
key = random.PRNGKey(456)
P_test = 100
Nx = m
keys = random.split(key, N_test)

config.update("jax_enable_x64", True)
u_test, y_test, s_test = vmap(generate_one_test_data, (0, None))(keys, P_test)

# Reshape Data
u_test = np.float32(u_test.reshape(N_test * P_test ** 2, -1))
y_test = np.float32(y_test.reshape(N_test * P_test ** 2, -1))
s_test = np.float32(s_test.reshape(N_test * P_test ** 2, -1))

# Predict
params = model.get_params(model.opt_state)
s_pred = model.predict_s(params, u_test, y_test)

# Generate an uniform mesh
x = np.linspace(0, 1, Nx)
t = np.linspace(0, 1, Nt)
XX, TT = np.meshgrid(x, t)

# Grid data
S_pred = griddata(y_test, s_pred.flatten(), (XX, TT), method='cubic')
S_test = griddata(y_test, s_test.flatten(), (XX, TT), method='cubic')

# Real
plot(XX, TT, S_test)
# Prediction
plot(XX, TT, S_pred)

# Compute the Error
error_s = np.linalg.norm(S_test - S_pred) / np.linalg.norm(S_test)
print(error_s)
