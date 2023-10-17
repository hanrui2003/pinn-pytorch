# -*- coding: utf-8 -*-
"""
Created on Wed Dec 8 2021

@author: Askeladd
"""
import sys
import time
import numpy as np
import torch

import utils
import rfm_s2t

from equations.membrane import Membrane

torch.set_default_dtype(torch.float64)

eqn = Membrane()

vanal_u = eqn.vanal_u
vanal_phi = eqn.vanal_phi
vanal_psi = eqn.vanal_psi
vanal_f = eqn.vanal_f


def cal_matrix(models,points,Nx,Ny,Nt,M,Qx,Qy,Qt,initial_u=None,initial_ut=None,tshift=0):

    # matrix define (Aw=b)
    Ae = np.zeros([Nx*Ny*Nt*Qx*Qy*Qt,Nx*Ny*Nt*M]) # u_tt - a^2 u_xx = 0
    fe = np.zeros([Nx*Ny*Nt*Qx*Qy*Qt,1])
    
    Ai_u = np.zeros([Nx*Ny*Qx*Qy,Nx*Ny*Nt*M]) # u(x,y,0) = phi(x,y) on t=0
    fi_u = np.zeros([Nx*Ny*Qx*Qy,1])
    
    Ai_ut = np.zeros([Nx*Ny*Qx*Qy,Nx*Ny*Nt*M]) # u_t(x,y,0) = psi(x,y) on t=0
    fi_ut = np.zeros([Nx*Ny*Qx*Qy,1])
    
    Ab_x0 = np.zeros([Ny*Nt*Qy*Qt,Nx*Ny*Nt*M]) # u(0,y,t) = 0
    fb_x0 = np.zeros([Ny*Nt*Qy*Qt,1])
    
    Ab_x1 = np.zeros([Ny*Nt*Qy*Qt,Nx*Ny*Nt*M]) # u(lx,y,t) = 0
    fb_x1 = np.zeros([Ny*Nt*Qy*Qt,1])
    
    Ab_y0 = np.zeros([Nx*Nt*Qx*Qt,Nx*Ny*Nt*M]) # u(x,0,t) = 0
    fb_y0 = np.zeros([Nx*Nt*Qx*Qt,1])
    
    Ab_y1 = np.zeros([Nx*Nt*Qx*Qt,Nx*Ny*Nt*M]) # u(x,ly,t) = 0
    fb_y1 = np.zeros([Nx*Nt*Qx*Qt,1])

    Ac_x0 = np.zeros([(Nx - 1)*Ny*Nt*Qy*Qt, Nx*Ny*Nt*M]) # C^0 continuity on x
    fc_x0 = np.zeros([(Nx - 1)*Ny*Nt*Qy*Qt, 1])

    Ac_x1 = np.zeros([(Nx - 1)*Ny*Nt*Qy*Qt, Nx*Ny*Nt*M]) # C^1 continuity on x
    fc_x1 = np.zeros([(Nx - 1)*Ny*Nt*Qy*Qt, 1])

    Ac_y0 = np.zeros([(Ny - 1)*Nx*Nt*Qx*Qt, Nx*Ny*Nt*M]) # C^0 continuity on y
    fc_y0 = np.zeros([(Ny - 1)*Nx*Nt*Qx*Qt, 1])

    Ac_y1 = np.zeros([(Ny - 1)*Nx*Nt*Qx*Qt, Nx*Ny*Nt*M]) # C^1 continuity on y
    fc_y1 = np.zeros([(Ny - 1)*Nx*Nt*Qx*Qt, 1])

    Ac_t0 = np.zeros([(Nt - 1)*Nx*Ny*Qx*Qy, Nx*Ny*Nt*M]) # C^0 continuity on t
    fc_t0 = np.zeros([(Nt - 1)*Nx*Ny*Qx*Qy, 1])

    Ac_t1 = np.zeros([(Nt - 1)*Nx*Ny*Qx*Qy, Nx*Ny*Nt*M]) # C^1 continuity on t
    fc_t1 = np.zeros([(Nt - 1)*Nx*Ny*Qx*Qy, 1])
    
    for kx in range(Nx):
        for ky in range(Ny):
            for n in range(Nt):

                in_ = points[kx][ky][n].detach().numpy()

                if n == 0:
                    if initial_u is None:
                        fi_u[(kx * Ny + ky) * Qx * Qy : (kx * Ny + ky + 1) * Qx * Qy,:] = vanal_phi(in_[:Qx, :Qy, 0, 0], in_[:Qx, :Qy, 0, 1]).reshape((Qx,1))
                    else:
                        fi_u[(kx * Ny + ky) * Qx * Qy : (kx * Ny + ky + 1) * Qx * Qy,:] = initial_u[kx * Qx : (kx + 1) * Qx, ky * Qy : (ky + 1) * Qy, :].reshape(-1, 1)

                    if initial_ut is None:
                        fi_ut[(kx * Ny + ky) * Qx * Qy : (kx * Ny + ky + 1) * Qx * Qy,:] = vanal_psi(in_[:Qx, :Qy, 0, 0], in_[:Qx, :Qy, 0, 1]).reshape((Qx,1))
                    else:
                        fi_ut[(kx * Ny + ky) * Qx * Qy : (kx * Ny + ky + 1) * Qx * Qy,:] = initial_ut[kx * Qx : (kx + 1) * Qx, ky * Qy : (ky + 1) * Qy, :].reshape(-1, 1)
                
                f_in = in_[:Qx,:Qy,:Qt,:].reshape((-1, 3))
                fe[(kx * Ny * Nt + ky * Nt + n) * Qx * Qy * Qt : (kx * Ny * Nt + ky * Nt + n + 1) * Qx * Qy * Qt, :] = vanal_f(f_in[:,0],f_in[:,1],f_in[:,2] + tshift).reshape(-1,1)
                
                # u_tt - a^2 (u_xx + u_yy) = 0

                out = models[kx][ky][n](points[kx][ky][n])
                values = out.detach().numpy()
                M_begin = (kx * Nt * Ny + ky * Nt + n) * M

                grads = []
                grads_2_xx = []
                grads_2_yy = []
                grads_2_tt = []

                for i in range(M):
                    g_1 = torch.autograd.grad(outputs=out[:,:,:,i], inputs=points[kx][ky][n],
                                            grad_outputs=torch.ones_like(out[:,:,:,i]),
                                            create_graph = True, retain_graph = True)[0]
                    grads.append(g_1.squeeze().detach().numpy())
                    
                    g_2_x = torch.autograd.grad(outputs=g_1[:,:,:,0], inputs=points[kx][ky][n],
                                        grad_outputs=torch.ones_like(out[:,:,:,i]),
                                        create_graph = True, retain_graph = True)[0]
                    
                    g_2_y = torch.autograd.grad(outputs=g_1[:,:,:,1], inputs=points[kx][ky][n],
                                        grad_outputs=torch.ones_like(out[:,:,:,i]),
                                        create_graph = True, retain_graph = True)[0]

                    g_2_t = torch.autograd.grad(outputs=g_1[:,:,:,2], inputs=points[kx][ky][n],
                                        grad_outputs=torch.ones_like(out[:,:,:,i]),
                                        create_graph = True, retain_graph = True)[0]

                    grads_2_xx.append(g_2_x[:,:,:,0].squeeze().detach().numpy())
                    grads_2_yy.append(g_2_y[:,:,:,1].squeeze().detach().numpy())
                    grads_2_tt.append(g_2_t[:,:,:,2].squeeze().detach().numpy())
                    
                grads = np.array(grads).swapaxes(0,4)
                
                # print(values.shape,grads.shape)

                grads_2_xx = np.array(grads_2_xx)
                grads_2_xx = grads_2_xx[:,:Qx,:Qy,:Qt]
                grads_2_xx = grads_2_xx.transpose(1,2,3,0).reshape(-1,M)

                grads_2_yy = np.array(grads_2_yy)
                grads_2_yy = grads_2_yy[:,:Qx,:Qy,:Qt]
                grads_2_yy = grads_2_yy.transpose(1,2,3,0).reshape(-1,M)
                
                grads_2_tt = np.array(grads_2_tt)
                grads_2_tt = grads_2_tt[:,:Qx,:Qy,:Qt]
                grads_2_tt = grads_2_tt.transpose(1,2,3,0).reshape(-1,M)
                
                Ae[(kx * Ny * Nt + ky * Nt + n) * Qx * Qy * Qt : (kx * Ny * Nt + ky * Nt + n + 1) * Qx * Qy * Qt, M_begin : M_begin + M] = grads_2_tt - eqn.coef_A * eqn.coef_A * (grads_2_xx + grads_2_yy)
                
                if n == 0:
                    # u(x,y,0) = ..
                    Ai_u[(kx * Ny + ky) * Qx * Qy : (kx * Ny + ky + 1) * Qx * Qy, M_begin : M_begin + M] = values[:Qx,: Qy, 0,:].reshape(-1, M)
                
                    # u_t(x,y,0) = ..    
                    Ai_ut[(kx * Ny + ky) * Qx * Qy : (kx * Ny + ky + 1) * Qx * Qy, M_begin : M_begin + M] = grads[2, :Qx, :Qy, 0,:].reshape(-1, M)

                if kx == 0:
                    # u(0,y,t) = 0
                    Ab_x0[(ky * Nt + n) * Qy * Qt : (ky * Nt + n + 1) * Qy * Qt, M_begin: M_begin + M] = values[0, :Qy, :Qt, :].reshape(-1, M)

                if kx == Nx - 1:
                    # u(lx,y,t) = 0
                    Ab_x1[(ky * Nt + n) * Qy * Qt : (ky * Nt + n + 1) * Qy * Qt, M_begin: M_begin + M] = values[-1, :Qy, :Qt, :].reshape(-1, M)

                if ky == 0:
                    # u(x,0,t) = 0
                    Ab_y0[(kx * Nt + n) * Qx * Qt : (kx * Nt + n + 1) * Qx * Qt, M_begin: M_begin + M] = values[:Qx, 0, :Qt, :].reshape(-1, M)

                if ky == Ny - 1:
                    # u(x,ly,t) = 0
                    Ab_y1[(kx * Nt + n) * Qx * Qt : (kx * Nt + n + 1) * Qx * Qt, M_begin: M_begin + M] = values[:Qx, -1, :Qt, :].reshape(-1, M)

                # C0 continuity on t
                if n > 0:
                    Ac_t0[(n - 1)*Nx*Ny*Qx*Qy + (kx*Ny+ky)*Qx*Qy : (n - 1)*Nx*Ny*Qx*Qy + (kx*Ny+ky+1)*Qx*Qy , M_begin: M_begin + M] = values[:Qx, :Qy, 0, :].reshape(-1, M)
                if n < Nt - 1:
                    Ac_t0[n*Nx*Ny*Qx*Qy + (kx*Ny+ky)*Qx*Qy : n*Nx*Ny*Qx*Qy + (kx*Ny+ky+1)*Qx*Qy, M_begin: M_begin + M] = -values[:Qx, :Qy, -1, :].reshape(-1, M)

                # C0 continuity on x
                if kx > 0:
                    Ac_x0[(kx - 1)*Ny*Nt*Qy*Qt + (ky*Nt+n)*Qy*Qt : (kx - 1)*Ny*Nt*Qy*Qt + (ky*Nt+n+1)*Qy*Qt , M_begin: M_begin + M] = values[0, :Qy, :Qt, :].reshape(-1, M)
                if kx < Nx - 1:
                    Ac_x0[kx*Ny*Nt*Qy*Qt + (ky*Nt+n)*Qy*Qt : kx*Ny*Nt*Qy*Qt + (ky*Nt+n+1)*Qy*Qt, M_begin: M_begin + M] = -values[-1, :Qy, :Qt, :].reshape(-1, M)

                # C0 continuity on y
                if ky > 0:
                    Ac_y0[(ky - 1)*Nx*Nt*Qx*Qt + (kx*Nt+n)*Qx*Qt : (ky - 1)*Nx*Nt*Qx*Qt + (kx*Nt+n+1)*Qx*Qt , M_begin: M_begin + M] = values[:Qx, 0, :Qt, :].reshape(-1, M)
                if ky < Ny - 1:
                    Ac_y0[ky*Nx*Nt*Qx*Qt + (kx*Nt+n)*Qx*Qt : ky*Nx*Nt*Qx*Qt + (kx*Nt+n+1)*Qx*Qt, M_begin: M_begin + M] = -values[:Qx, -1, :Qt, :].reshape(-1, M)

                # C1 continuity on t
                if n > 0:
                    Ac_t1[(n - 1)*Nx*Ny*Qx*Qy + (kx*Ny+ky)*Qx*Qy : (n - 1)*Nx*Ny*Qx*Qy + (kx*Ny+ky+1)*Qx*Qy , M_begin: M_begin + M] = grads[2, :Qx, :Qy, 0, :].reshape(-1, M)
                if n < Nt - 1:
                    Ac_t1[n*Nx*Ny*Qx*Qy + (kx*Ny+ky)*Qx*Qy : n*Nx*Ny*Qx*Qy + (kx*Ny+ky+1)*Qx*Qy, M_begin: M_begin + M] = -grads[2, :Qx, :Qy, -1, :].reshape(-1, M)

                # C1 continuity on x
                if kx > 0:
                    Ac_x1[(kx - 1)*Ny*Nt*Qy*Qt + (ky*Nt+n)*Qy*Qt : (kx - 1)*Ny*Nt*Qy*Qt + (ky*Nt+n+1)*Qy*Qt , M_begin: M_begin + M] = grads[0, 0, :Qy, :Qt, :].reshape(-1, M)
                if kx < Nx - 1:
                    Ac_x1[kx*Ny*Nt*Qy*Qt + (ky*Nt+n)*Qy*Qt : kx*Ny*Nt*Qy*Qt + (ky*Nt+n+1)*Qy*Qt, M_begin: M_begin + M] = -grads[0, -1, :Qy, :Qt, :].reshape(-1, M)

                # C1 continuity on y
                if ky > 0:
                    Ac_y1[(ky - 1)*Nx*Nt*Qx*Qt + (kx*Nt+n)*Qx*Qt : (ky - 1)*Nx*Nt*Qx*Qt + (kx*Nt+n+1)*Qx*Qt , M_begin: M_begin + M] = grads[1, :Qx, 0, :Qt, :].reshape(-1, M)
                if ky < Ny - 1:
                    Ac_y1[ky*Nx*Nt*Qx*Qt + (kx*Nt+n)*Qx*Qt : ky*Nx*Nt*Qx*Qt + (kx*Nt+n+1)*Qx*Qt, M_begin: M_begin + M] = -grads[1, :Qx, -1, :Qt, :].reshape(-1, M)


    A = np.concatenate((Ae, Ai_u, Ai_ut, Ab_x0, Ab_x1, Ab_y0, Ab_y1, Ac_x0, Ac_y0, Ac_t0, Ac_x1, Ac_y1, Ac_t1,), axis=0)
    f = np.concatenate((fe, fi_u, fi_ut, fb_x0, fb_x1, fb_y0, fb_y1, fc_x0, fc_y0, fc_t0, fc_x1, fc_y1, fc_t1,), axis=0)
    
    return(A,f)


def main(Nx,Ny,Nt,M,Qx,Qy,Qt,time_block=1,plot = False,moore = False):

    # prepare models and collocation pointss

    tlen = tf / time_block

    models, points = rfm_s2t.pre_define_rfm(Nx=Nx, Ny=Ny, Nt=Nt, M=M, Qx=Qx, Qy=Qy, Qt=Qt, x0=eqn.x0, x1=eqn.x1, y0=eqn.y0, y1=eqn.y1, t0=0, t1=tlen)

    initial_u = rfm_s2t.get_anal_u(vanal_u=vanal_phi, points=points, Nx=Nx, Ny=Ny, Qx=Qx, Qy=Qy, nt=0, qt=0)
    initial_ut = rfm_s2t.get_anal_u(vanal_u=vanal_psi, points=points, Nx=Nx, Ny=Ny, Qx=Qx, Qy=Qy, nt=0, qt=0)

    true_values = list()
    numerical_values = list()
    L_is = list()
    L_2s = list()
    e_i = list()
    e_2 = list()

    for t in range(time_block):
    
        # matrix define (Aw=b)
        A,f = cal_matrix(models,points,Nx,Ny,Nt,M,Qx,Qy,Qt,initial_u=initial_u,initial_ut=initial_ut,tshift=t * tlen)

        w = rfm_s2t.solve_lst_square(A, f, moore=moore)

        final_u = rfm_s2t.get_num_u(models=models, points=points, w=w, Nx=Nx, Ny=Ny, Nt=Nt, M=M, Qx=Qx, Qy=Qy, Qt=Qt, nt=Nt-1, qt=Qt)
        final_ut = rfm_s2t.get_num_ut(models=models, points=points, w=w, Nx=Nx, Ny=Ny, Nt=Nt, M=M, Qx=Qx, Qy=Qy, Qt=Qt, nt=Nt-1, qt=Qt)

        initial_u = final_u
        initial_ut = final_ut

        true_values_, numerical_values_, L_i, L_2 = rfm_s2t.test(vanal_u=vanal_u, models=models, \
            Nx=Nx, Ny=Ny, Nt=Nt, M=M, Qx=Qx, Qy=Qy, Qt=Qt, \
            w=w, \
            x0=eqn.x0, x1=eqn.x1, y0=eqn.y0, y1=eqn.y1, t0=0, t1=tlen, block=t)

        true_values.append(true_values_)
        numerical_values.append(numerical_values_)
        L_is.append(L_i)
        L_2s.append(L_2)

        t_unit = true_values_.shape[2] // Nt

        true_slices = np.concatenate([true_values_[:, :, ::t_unit], true_values_[:, :, -1:]], axis=2)
        nume_slices = np.concatenate([numerical_values_[:, :, ::t_unit], numerical_values_[:, :, -1:]], axis=2)

        e_i.append(utils.L_inf_error(true_slices - nume_slices, axis=(0, 1)))
        e_2.append(utils.L_2_error(true_slices - nume_slices, axis=(0, 1)))

    true_values = np.concatenate(true_values, axis=2)#[:, ::time_block]
    numerical_values = np.concatenate(numerical_values, axis=2)#[:, ::time_block]

    L_is = np.array(L_is)
    L_2s = np.array(L_2s)
    e_i = np.array(e_i)
    e_2 = np.array(e_2)

    print(L_is.shape, L_2s.shape, e_i.shape, e_2.shape)
    
    # visualize
    if plot:
        # utils.visualize_2d(true_values, numerical_values, x0=eqn.x0, x1=eqn.x1, y0=eqn.y0, y1=eqn.y1, t0=0, t1=tf, eqname="membrane-2", mename=f"rfm-time_block={time_block:d}-Nx={Nx:d}-Ny={Ny:d}-Nt={Nt:d}-M={M:d}-Qx={Qx:d}-Qy={Qy:d}-Qt={Qt:d}")
        utils.visualize_3d_wo(eqn, true_values, x0=eqn.x0, x1=eqn.x1, y0=eqn.y0, y1=eqn.y1, t0=0, t1=tf, savedir=f"outputs/{eqn.name}_rfm_tf={tf:.2f}_time_block={time_block:d}-Nx={Nx:d}-Ny={Ny:d}-Nt={Nt:d}-M={M:d}-Qx={Qx:d}-Qy={Qy:d}-Qt={Qt:d}/trusol")
        utils.visualize_3d_wo(eqn, numerical_values, x0=eqn.x0, x1=eqn.x1, y0=eqn.y0, y1=eqn.y1, t0=0, t1=tf, savedir=f"outputs/{eqn.name}_rfm_tf={tf:.2f}_time_block={time_block:d}-Nx={Nx:d}-Ny={Ny:d}-Nt={Nt:d}-M={M:d}-Qx={Qx:d}-Qy={Qy:d}-Qt={Qt:d}/numsol")
        utils.visualize_3d_wo(eqn, np.abs(true_values - numerical_values), x0=eqn.x0, x1=eqn.x1, y0=eqn.y0, y1=eqn.y1, t0=0, t1=tf, savedir=f"outputs/{eqn.name}_rfm_tf={tf:.2f}_time_block={time_block:d}-Nx={Nx:d}-Ny={Ny:d}-Nt={Nt:d}-M={M:d}-Qx={Qx:d}-Qy={Qy:d}-Qt={Qt:d}/epsilon")

    return L_is, L_2s, e_i, e_2



if __name__ == '__main__':
    # set_seed(100)

    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 1

    if len(sys.argv) > 2:
        tf = float(sys.argv[2])
    else:
        tf = 10.0


    # time_blocks = [5,]
    # Nxs = [2,]
    # Nys = [2,]
    # Nts = [2,]
    # Ms = [400,]
    # Qxs = [10,]
    # Qys = [10,]
    # Qts = [10,]
    time_blocks = [1,]
    Nxs = [1,]
    Nys = [1,]
    Nts = [1,]
    Ms = [100,]
    Qxs = [10,]
    Qys = [10,]
    Qts = [10,]

    for i, (Nx, Ny, Nt, M, Qx, Qy, Qt, time_block) in enumerate(zip(Nxs, Nys, Nts, Ms, Qxs, Qys, Qts, time_blocks)):
        L_i_res = list()
        L_2_res = list()
        e_i_res = list()
        e_2_res = list()
        result = list()
        for k in range(n):
            print(f"[Iter={k+1:d}/{n:d}] Nx={Nx:d}, Ny={Ny:d}, Nt={Nt:d}, M={M:d}, Qx={Qx:d}, Qy={Qy:d}, Qt={Qt:d}, time_block={time_block:d}")
            plot = (k == 0)
            L_is, L_2s, e_i, e_2 = main(Nx=Nx,Ny=Ny,Nt=Nt,M=M,Qx=Qx,Qy=Qy,Qt=Qt,time_block=time_block,plot=plot)

            L_i_res.append(L_is)
            L_2_res.append(L_2s)
            e_i_res.append(e_i)
            e_2_res.append(e_2)

            L_i = utils.L_inf_error(L_is)
            L_2 = utils.L_2_error(L_2s)
            
            result.append([L_i, L_2])

        L_i_res = np.array(L_i_res)
        L_2_res = np.array(L_2_res)
        e_i_res = np.array(e_i_res).reshape(n, time_block, Nt + 1)
        e_2_res = np.array(e_2_res).reshape(n, time_block, Nt + 1)
        result = np.array(result)

        if i == 0:
            fm = "a"
        else:
            fm = "a"

        with open(f"logs/trial-twod_rfm_{eqn.name}-tf{tf:.2e}.log", fm) as f:

            print(f"Nx={Nx:d}, Ny={Ny:d}, Nt={Nt:d}, M={M:d}, Qx={Qx:d}, Qy={Qy:d}, Qt={Qt:d}, t_block={time_block:d}", file=f)
            print("Mean L_i: {:.2e}, Max L_i: {:.2e}".format(result[:, 0].mean(), result[:, 0].max()), file=f)
            print("Mean L_2: {:.2e}, Max L_2: {:.2e}".format(result[:, 1].mean(), result[:, 1].max()), file=f)
            
            print("-" * 20, file=f)

            np.set_printoptions(formatter={'float': '{:.2e}'.format})

            print(f"Mean L_i per block: {L_i_res.mean(0)}", file=f)
            print(f"Max L_i per block: {L_i_res.max(0)}", file=f)
            print(f"Mean L_2 per block: {L_2_res.mean(0)}", file=f)
            print(f"Max L_2 per block: {L_2_res.max(0)}", file=f)
            
            print("-" * 20, file=f)

            for tb in range(time_block):
                e_i_ = e_i_res[:, tb, :]
                e_2_ = e_2_res[:, tb, :]
                print(f"Max L_i on block={tb:d}: {e_i_.max(0)}", file=f)
                print(f"Max L_2 on block={tb:d}: {e_2_.max(0)}", file=f)
            
            print("#" * 40, file=f)
            
