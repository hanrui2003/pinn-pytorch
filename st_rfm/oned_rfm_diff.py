# -*- coding: utf-8 -*-
"""
Created on Wed Dec 8 2021

@author: Askeladd
"""
import os
import sys
import time
import numpy as np
import torch

import utils
import rfm

from equations.diffusion import Diffusion

torch.set_default_dtype(torch.float64)

eqn = Diffusion()

vanal_u = eqn.vanal_u
vanal_f = eqn.vanal_f


def cal_matrix(models, points, Nx, Nt, M, Qx, Qt, initial=None, tshift=0):
    # matrix define (Aw=b)
    # pde配点
    A_1 = np.zeros([Nx * Nt * Qx * Qt, Nx * Nt * M])  # u_t - lapacian u - u^2 = f(x, t) on omega [0, T]
    f_1 = np.zeros([Nx * Nt * Qx * Qt, 1])

    # 左边界配点
    A_2 = np.zeros([Nt * Qt, Nx * Nt * M])  # u(x,t) = g(x, t) on x=0
    f_2 = np.zeros([Nt * Qt, 1])

    # 右边界配点
    A_3 = np.zeros([Nt * Qt, Nx * Nt * M])  # u(x,t) = g(x, t) on x=1
    f_3 = np.zeros([Nt * Qt, 1])

    # 初值配点
    A_4 = np.zeros([Nx * Qx, Nx * Nt * M])  # u(x,t) = h(x) on t=0
    f_4 = np.zeros([Nx * Qx, 1])

    C_0x = np.zeros([(Nx - 1) * Qt * Nt, Nx * Nt * M])  # C^0 continuity on x
    f_c_0x = np.zeros([(Nx - 1) * Qt * Nt, 1])

    C_0t = np.zeros([(Nt - 1) * Qx * Nx, Nx * Nt * M])  # C^0 continuity on t
    f_c_0t = np.zeros([(Nt - 1) * Qx * Nx, 1])

    C_1x = np.zeros([(Nx - 1) * Qt * Nt, Nx * Nt * M])  # C^1 continuity on x
    f_c_1x = np.zeros([(Nx - 1) * Qt * Nt, 1])

    for k in range(Nx):
        for n in range(Nt):
            # print(k,n)
            # u_t - lapacian u - u^2 = f(x, t) on omega [0, T]
            in_ = points[k][n].detach().numpy()
            out = models[k][n](points[k][n])
            values = out.detach().numpy()
            M_begin = (k * Nt + n) * M

            # u(x, 0) = ...
            if n == 0:
                # A_4[k*Qx : (k+1)*Qx, M_begin : M_begin + M] = values[:Qx,0,:]
                if initial is None:
                    f_4[k * Qx: (k + 1) * Qx, :] = vanal_u(in_[:Qx, 0, 0], in_[:Qx, 0, 1]).reshape((Qx, 1))
                else:
                    f_4[k * Qx: (k + 1) * Qx, :] = initial[k * Qx: (k + 1) * Qx, :]

            # u(0,t) = ..
            if k == 0:
                # A_2[0, M_begin : M_begin + M] = values[0,:Qt,:]
                f_2[n * Qt: n * Qt + Qt, :] = \
                    vanal_u(in_[0, :Qt, 0], in_[0, :Qt, 1] + tshift).reshape((Qt, 1))
                # print(in_[0,:Qt,:])

            # u(1,t) = ..
            if k == Nx - 1:
                # A_5[n*Qt : n*Qt+Qt, M_begin : M_begin + M] = values[-1,:Qt,:]
                f_3[n * Qt: n * Qt + Qt, :] = \
                    vanal_u(in_[-1, :Qt, 0], in_[-1, :Qt, 1] + tshift).reshape((Qt, 1))
                # print(in_[-1,:Qt,:])

            # 转成二维点列，去除区域的右边界和上边界的点。然后计算方程的右端项f
            f_in = in_[:Qx, :Qt, :].reshape((-1, 2))
            f_1[k * Nt * Qx * Qt + n * Qx * Qt: k * Nt * Qx * Qt + n * Qx * Qt + Qx * Qt, :] = vanal_f(f_in[:, 0],
                                                                                                       f_in[:,
                                                                                                       1] + tshift).reshape(
                -1, 1)

            # u_t - lapacian u - u^2 = f(x, t) on omega [0, T]
            grads = []
            grads_2_xx = []
            grads_2_yy = []
            for i in range(M):
                g_1 = torch.autograd.grad(outputs=out[:, :, i], inputs=points[k][n],
                                          grad_outputs=torch.ones_like(out[:, :, i]),
                                          create_graph=True, retain_graph=True)[0]
                grads.append(g_1.squeeze().detach().numpy())

                g_2_x = torch.autograd.grad(outputs=g_1[:, :, 0], inputs=points[k][n],
                                            grad_outputs=torch.ones_like(out[:, :, i]),
                                            create_graph=True, retain_graph=True)[0]
                g_2_y = torch.autograd.grad(outputs=g_1[:, :, 1], inputs=points[k][n],
                                            grad_outputs=torch.ones_like(out[:, :, i]),
                                            create_graph=True, retain_graph=True)[0]
                grads_2_xx.append(g_2_x[:, :, 0].squeeze().detach().numpy())
                grads_2_yy.append(g_2_y[:, :, 1].squeeze().detach().numpy())

            grads = np.array(grads).swapaxes(0, 3)

            # print(values.shape,grads.shape)

            grads_t = grads[1, :Qx, :Qt, :]
            grads_t = grads_t.reshape(-1, M)
            grads_2_xx = np.array(grads_2_xx)
            grads_2_xx = grads_2_xx[:, :Qx, :Qt]
            grads_2_xx = grads_2_xx.transpose(1, 2, 0).reshape(-1, M)

            A_1[k * Nt * Qx * Qt + n * Qx * Qt: k * Nt * Qx * Qt + n * Qx * Qt + Qx * Qt,
            M_begin: M_begin + M] = grads_t - eqn.nu * grads_2_xx  # - np.power(values[:Qx, :Qt, :], 2.0).reshape(-1, M)

            # u(0,t) = ..
            if k == 0:
                A_2[n * Qt: n * Qt + Qt, M_begin: M_begin + M] = values[0, :Qt, :]
            # u(1,t) = ..
            if k == Nx - 1:
                A_3[n * Qt: n * Qt + Qt, M_begin: M_begin + M] = values[-1, :Qt, :]
            # u(x,0) = ..
            if n == 0:
                A_4[k * Qx: k * Qx + Qx, M_begin: M_begin + M] = values[:Qx, 0, :]

            # C0 continuity on t
            if n > 0:
                C_0t[(n - 1) * Qx * Nx + k * Qx: (n - 1) * Qx * Nx + k * Qx + Qx, M_begin: M_begin + M] = values[:Qx, 0,
                                                                                                          :]
            if n < Nt - 1:
                C_0t[n * Qx * Nx + k * Qx: n * Qx * Nx + k * Qx + Qx, M_begin: M_begin + M] = -values[:Qx, -1, :]

            # C0 continuity on x
            if k > 0:
                C_0x[(k - 1) * Qt * Nt + n * Qt: (k - 1) * Qt * Nt + (n + 1) * Qt, M_begin: M_begin + M] = values[0,
                                                                                                           :Qt, :]
            if k < Nx - 1:
                C_0x[k * Nt * Qt + n * Qt: k * Nt * Qt + (n + 1) * Qt, M_begin: M_begin + M] = -values[-1, :Qt, :]

            # C1 continuity on x
            if k > 0:
                C_1x[(k - 1) * Qt * Nt + n * Qt: (k - 1) * Qt * Nt + (n + 1) * Qt, M_begin: M_begin + M] = grads[0, 0,
                                                                                                           :Qt, :]
            if k < Nx - 1:
                C_1x[k * Nt * Qt + n * Qt: k * Nt * Qt + (n + 1) * Qt, M_begin: M_begin + M] = -grads[0, -1, :Qt, :]

    A = np.concatenate((A_1, A_2, A_3, A_4, C_0t, C_0x, C_1x), axis=0)
    f = np.concatenate((f_1, f_2, f_3, f_4, f_c_0t, f_c_0x, f_c_1x), axis=0)
    # print(f.shape)
    return (A, f)


def main(Nx, Nt, M, Qx, Qt, tf=1.0, time_block=1, plot=False, moore=False, img_save_dir=None):
    # prepare models and collocation pointss

    tlen = tf / time_block

    models, points = rfm.pre_define_rfm(Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt, x0=eqn.a1, x1=eqn.b1, t0=0, t1=tlen)

    initial = rfm.get_anal_u(vanal_u=vanal_u, points=points, Nx=Nx, Qx=Qx, nt=0, qt=0)

    true_values = list()
    numerical_values = list()
    L_is = list()
    L_2s = list()

    for t in range(time_block):
        # matrix define (Aw=b)
        A, f = cal_matrix(models, points, Nx, Nt, M, Qx, Qt, initial=initial, tshift=t * tlen)

        w = rfm.solve_lst_square(A, f, moore=moore)

        # 当前time block的终值
        final = rfm.get_num_u(models=models, points=points, w=w, Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt, nt=Nt - 1, qt=Qt)
        initial = final
        true_values_, numerical_values_, L_i, L_2 = utils.test(vanal_u=vanal_u, models=models, \
                                                               Nx=Nx, Nt=Nt, M=M, Qx=Qx, Qt=Qt, \
                                                               w=w, \
                                                               x0=eqn.a1, x1=eqn.b1, t0=0, t1=tlen, block=t)

        true_values.append(true_values_)
        numerical_values.append(numerical_values_)
        L_is.append(L_i)
        L_2s.append(L_2)

    true_values = np.concatenate(true_values, axis=1)[:, ::time_block]
    numerical_values = np.concatenate(numerical_values, axis=1)[:, ::time_block]
    e_i = utils.L_inf_error(true_values - numerical_values, axis=0).reshape(time_block, Nt, -1)
    e_2 = utils.L_2_error(true_values - numerical_values, axis=0).reshape(time_block, Nt, -1)
    L_is = np.array(L_is)
    L_2s = np.array(L_2s)

    # visualize
    if plot:
        utils.visualize(true_values, numerical_values, x0=eqn.a1, x1=eqn.b1, t0=0, t1=tf, savedir=img_save_dir,
                        eqname=eqn.name, mename="rfm")

    return L_is, L_2s, e_i, e_2


if __name__ == '__main__':
    # utils.set_seed(100)

    # time_block = 10

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 1

    # time final
    if len(sys.argv) > 2:
        tf = float(sys.argv[2])
    else:
        tf = 1.0

    # 以下是五组配置，每次训练只取一列
    time_blocks = [ 1, ]
    # x维度划分的区间数
    Nxs = [ 5, ]
    # t维度划分的区间数
    Nts = [ 5, ]
    # 每个局部局域的特征函数数量
    Ms = [ 300, ]
    # x维度每个区间的配点数，Qx+1
    Qxs = [ 30, ]
    # t维度每个区间的配点数，Qt+1
    Qts = [ 30, ]

    utils.record(main, eqn.name, recursive_times=n, tf=tf, time_blocks=time_blocks, Nxs=Nxs, Nts=Nts, Ms=Ms, Qxs=Qxs,
                 Qts=Qts, eqn=eqn.name, method="rfm")
