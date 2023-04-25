import jax
import jax.numpy as jnp
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

import numpy as np

from scipy.interpolate import griddata

"""Solve 1D
    u_t = (k(x) u_x)_x - v(x) u_x + g(u) + f(x)
    with zero initial and boundary conditions.
"""


def plot(X, T, f):
    fig, ax = plt.subplots(1, 1)
    cp = ax.contourf(X, T, f.T, 20, cmap="rainbow")
    fig.colorbar(cp)  # Add a colorbar to a plot
    ax.set_title('F(x,t)')
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    plt.show()

    ax = plt.axes(projection='3d')
    ax.plot_surface(T, X, f.T, cmap="rainbow")
    ax.set_xlabel('t')
    ax.set_ylabel('x')
    ax.set_zlabel('f(x,t)')
    plt.show()


# Use double precision to generate data (due to GP sampling)
def RBF_JAX(x1, x2, params):
    output_scale, lengthscales = params
    diffs = jnp.expand_dims(x1 / lengthscales, 1) - jnp.expand_dims(x2 / lengthscales, 0)
    r2 = jnp.sum(diffs ** 2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)


def RBF(x1, x2, params):
    output_scale, lengthscales = params
    diffs = np.expand_dims(x1 / lengthscales, 1) - np.expand_dims(x2 / lengthscales, 0)
    r2 = np.sum(diffs ** 2, axis=2)
    return output_scale * jnp.exp(-0.5 * r2)


if "__main__" == __name__:
    config.update("jax_enable_x64", True)
    xmin, xmax = 0, 1
    tmin, tmax = 0, 1
    # 函数：返回与输入同形的向量（或标量），每个元素为0.01
    k = lambda x: 0.01 * jnp.ones_like(x)
    # 零函数：返回与输入同形的零向量（或标量）
    v = lambda x: jnp.zeros_like(x)
    # 函数g(u)=0.01*u^2
    g = lambda u: 0.01 * u ** 2
    # 导函数g'(u) = 0.02*u
    dg = lambda u: 0.02 * u
    # 零函数：返回与输入同形的零向量（或标量）
    u0 = lambda x: jnp.zeros_like(x)

    # Generate subkeys
    # JAX的key不能重用，不然每次生成的都一样，所以每次要用随机数的时候，都要生成新的子key
    # 以下代码是分成两个新的key，用来生成随机数
    key = random.PRNGKey(0)
    subkeys = random.split(key, 2)

    # Generate a GP sample
    N = 500
    length_scale = 0.2
    gp_params = (1.0, length_scale)
    jitter = 1e-10
    X = jnp.linspace(xmin, xmax, N)[:, None]
    K = RBF_JAX(X, X, gp_params)
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(N))
    gp_sample = jnp.dot(L, random.normal(subkeys[0], (N,)))
    # Create a callable interpolation function
    # 一维分段线性差值
    # 和numpy的interp差不多，第二第三个参数是用来做差值函数的横纵坐标，第一个参数是要进行估计的x值
    f_fn = lambda x: jnp.interp(x, X.flatten(), gp_sample)

    # Create grid
    Nx = 50
    Nt = 100
    x = jnp.linspace(xmin, xmax, Nx)
    t = jnp.linspace(tmin, tmax, Nt)
    h = x[1] - x[0]
    dt = t[1] - t[0]
    h2 = h ** 2

    # Compute coefficients and forcing
    k = k(x)
    v = v(x)
    f = f_fn(x)

    # Compute finite difference operators
    D1 = jnp.eye(Nx, k=1) - jnp.eye(Nx, k=-1)
    D2 = -2 * jnp.eye(Nx) + jnp.eye(Nx, k=-1) + jnp.eye(Nx, k=1)
    D3 = jnp.eye(Nx - 2)
    M = -jnp.diag(D1 @ k) @ D1 - 4 * jnp.diag(k) @ D2
    m_bond = 8 * h2 / dt * D3 + M[1:-1, 1:-1]
    v_bond = 2 * h * jnp.diag(v[1:-1]) @ D1[1:-1, 1:-1] + 2 * h * jnp.diag(
        v[2:] - v[: Nx - 2]
    )
    mv_bond = m_bond + v_bond
    c = 8 * h2 / dt * D3 - M[1:-1, 1:-1] - v_bond

    # Initialize solution and apply initial condition
    u = jnp.zeros((Nx, Nt))
    u = u.at[:, 0].set(u0(x))


    # u = index_update(u, index[:,0], u0(x))
    # Time-stepping update
    def body_fn(i, u):
        gi = g(u[1:-1, i])
        dgi = dg(u[1:-1, i])
        h2dgi = jnp.diag(4 * h2 * dgi)
        A = mv_bond - h2dgi
        b1 = 8 * h2 * (0.5 * f[1:-1] + 0.5 * f[1:-1] + gi)
        b2 = (c - h2dgi) @ u[1:-1, i].T
        u = u.at[1:-1, i + 1].set(jnp.linalg.solve(A, b1 + b2))
        # u = index_update(u, index[1:-1, i + 1], np.linalg.solve(A, b1 + b2))
        return u


    # Run loop
    UU = lax.fori_loop(0, Nt - 1, body_fn, u)

    # Input sensor locations and measurements
    # xx = np.linspace(xmin, xmax, Nx)
    # u = f_fn(xx)
    # Output sensor locations and measurements
    # P = 300
    # idx = random.randint(subkeys[1], (P, 2), 0, max(Nx, Nt))
    # y = np.concatenate([x[idx[:, 0]][:, None], t[idx[:, 1]][:, None]], axis=1)
    # s = UU[idx[:, 0], idx[:, 1]]
    # x, t: sampled points on grid
    # return (x, t, UU), (u, y, s)

    XX, TT = jnp.meshgrid(x, t)
    plot(XX, TT, UU)

    # start my numerical solver
    # Generate a GP sample
    h = jax.device_get(h)
    dt = jax.device_get(dt)
    h2 = jax.device_get(h2)
    D2 = jax.device_get(D2)
    f = jax.device_get(f)

    D = 0.01
    k = 0.01
    print("grid ratio", D * dt / h2)

    u = np.zeros((Nx, Nt))
    for n in range(Nt - 1):
        u_n = u[:, n]
        u_next = u_n + dt * ((D / h2) * D2 @ u_n + k * u_n ** 2 + f)
        u_next[np.array([0, -1])] = 0.
        u[:, n + 1] = u_next

    plot(XX, TT, u)
