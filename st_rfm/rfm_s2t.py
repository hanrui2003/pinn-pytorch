# -*- coding: utf-8 -*-

from email.mime import base
import numpy as np
import torch
import torch.nn as nn
import scipy.sparse as scisparse
import itertools
from scipy.linalg import lstsq,pinv,svd
from scipy.sparse.linalg import svds
from scipy.fftpack import fftshift,fftn
import matplotlib.pyplot as plt

import utils

rand_mag = 1.0
def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        if m.weight is not None:
            nn.init.uniform_(m.weight, a = -rand_mag / 3, b = rand_mag / 3)
        if m.bias is not None:
            nn.init.uniform_(m.bias, a = -rand_mag, b = rand_mag)
        #nn.init.normal_(m.weight, mean=0, std=1)
        #nn.init.normal_(m.bias, mean=0, std=1)


class SinAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


class CosAct(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cos(x)

class local_rep(nn.Module):
    def __init__(self, in_features, out_features, hidden_layers, M, x_max, x_min, y_max, y_min, t_max, t_min):
        super(local_rep, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.hidden_features = M
        self.hidden_layers  = hidden_layers
        self.M = M
        self.a = torch.tensor([2.0/(x_max - x_min),2.0/(y_max - y_min),2.0/(t_max - t_min)])
        self.x_0 = torch.tensor([(x_max + x_min)/2,(y_max + y_min)/2,(t_max + t_min)/2])
        self.hidden_layer = nn.Sequential(nn.Linear(self.in_features, self.hidden_features, bias=True),nn.Tanh())

    def forward(self,x):
        x = self.a * (x - self.x_0)
        x = self.hidden_layer(x)
        return x


def pre_define_rfm(Nx,Ny,Nt,M,Qx,Qy,Qt,x0,x1,y0,y1,t0,t1):
    models = []
    points = []
    Lx = x1 - x0
    Ly = y1 - y0
    tf = t1 - t0
    for kx in range(Nx):
        
        x_min = Lx/Nx * kx + x0
        x_max = Lx/Nx * (kx+1) + x0
        model_for_x = list()
        point_for_x = list()
        x_devide = np.linspace(x_min, x_max, Qx + 1)

        for ky in range(Ny):
            
            y_min = Ly/Ny * ky + y0
            y_max = Ly/Ny * (ky+1) + y0
            model_for_xy = []
            point_for_xy = []
            y_devide = np.linspace(y_min, y_max, Qy + 1)

            for n in range(Nt):
                t_min = tf/Nt * n + t0
                t_max = tf/Nt * (n+1) + t0
                model = local_rep(in_features = 3, out_features = 1, hidden_layers = 1, M = M, x_min = x_min, 
                                x_max = x_max, y_min = y_min, y_max = y_max, t_min = t_min, t_max = t_max)
                model = model.apply(weights_init)
                model = model.double()
                for param in model.parameters():
                    param.requires_grad = False
                model_for_xy.append(model)
                t_devide = np.linspace(t_min, t_max, Qt + 1)
                grid = np.array(list(itertools.product(x_devide,y_devide,t_devide))).reshape(Qx+1,Qy+1,Qt+1,3)
                point_for_xy.append(torch.tensor(grid,requires_grad=True))
            model_for_x.append(model_for_xy)
            point_for_x.append(point_for_xy)
        models.append(model_for_x)
        points.append(point_for_x)

    return(models,points)


class local_mul_rep(nn.Module):
    def __init__(self, space_features, out_features, hidden_layers, Mx, Mt, x_max, x_min, y_max, y_min, t_max, t_min):
        super(local_mul_rep, self).__init__()
        self.x_features = space_features
        self.out_features = out_features
        self.hidden_features = Mx
        self.hidden_layers  = hidden_layers
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.t_max = t_max
        self.t_min = t_min
        self.Mx = Mx
        self.Mt = Mt
        self.a_x = torch.tensor([2.0 / (x_max - x_min), 2.0 / (y_max - y_min)])
        self.a_t = 2.0 / (t_max - t_min)
        self.x_0 = torch.tensor([(x_max + x_min) / 2.0, (y_max + y_min) / 2.0])
        self.t_0 = (t_max + t_min) / 2.0
        self.hx = nn.Sequential(nn.Linear(self.x_features, self.hidden_features, bias=True),nn.Tanh())
        self.ht = nn.Sequential(nn.Linear(1, self.hidden_features * self.Mt, bias=True),nn.Tanh())
        #print([x_min,x_max],[t_min,t_max])

    def forward(self,x):

        part = x.size()[:-1]
        
        # x-axis
        y = self.a_x * (x[..., :self.x_features] - self.x_0)
        y_x = self.hx(y)

        # t-axis
        y = self.a_t * (x[..., self.x_features:] - self.t_0)
        y_t_rfm = self.ht(y)

        c0 = (y >= -1) & (y <= 1)
        y0 = 1.0

        y_t_pou = c0 * y0

        y_t = y_t_pou * y_t_rfm
        
        y = y_x.view(*part, self.Mx, 1) * y_t.view(*part, -1, self.Mt)
        y = y.view(*part, -1)
        
        return y


class local_mul_t_psib_rep(nn.Module):
    def __init__(self, space_features, out_features, hidden_layers, Mx, Mt, x_max, x_min, y_max, y_min, t_max, t_min, t0, t1):
        super(local_mul_t_psib_rep, self).__init__()
        self.x_features = space_features
        self.out_features = out_features
        self.hidden_features = Mx
        self.hidden_layers  = hidden_layers
        self.x_max = x_max
        self.x_min = x_min
        self.y_max = y_max
        self.y_min = y_min
        self.t_max = t_max
        self.t_min = t_min
        self.Mx = Mx
        self.Mt = Mt
        self.t0 = t0
        self.t1 = t1
        self.a_x = torch.tensor([2.0 / (x_max - x_min), 2.0 / (y_max - y_min)])
        self.a_t = 2.0 / (t_max - t_min)
        self.x_0 = torch.tensor([(x_max + x_min) / 2.0, (y_max + y_min) / 2.0])
        self.t_0 = (t_max + t_min) / 2.0
        self.hx = nn.Sequential(nn.Linear(self.x_features, self.hidden_features, bias=True),nn.Tanh())
        self.ht = nn.Sequential(nn.Linear(1, self.hidden_features * self.Mt, bias=True),nn.Tanh())
        #print([x_min,x_max],[t_min,t_max])

    def forward(self,x):

        part = x.size()[:-1]
        
        # x-axis
        x_axis = self.a_x * (x[..., :self.x_features] - self.x_0)
        y_x = self.hx(x_axis)

        # t-axis
        t_axis = self.a_t * (x[..., self.x_features:] - self.t_0)

        y_rfm = self.ht(t_axis)
        
        c0 = (t_axis > -5 / 4) & (t_axis <= -3 / 4)
        c1 = (t_axis > -3 / 4) & (t_axis <= 3 / 4)
        c2 = (t_axis > 3 / 4) & (t_axis <= 5 / 4)

        if self.t_min == self.t0:
            y0 = 1.0
        else:
            y0 = (1 + torch.sin(2 * torch.pi * t_axis)) / 2
        
        y1 = 1.0
        
        if self.t_max == self.t1:
            y2 = 1.0
        else:
            y2 = (1 - torch.sin(2 * torch.pi * t_axis)) / 2

        y_pou = c0 * y0 + c1 * y1 + c2 * y2
        
        y_t = y_pou * y_rfm
        
        y = y_x.view(*part, self.Mx, 1) * y_t.view(*part, -1, self.Mt)
        y = y.view(*part, -1)
        
        return y


def pre_define_mul(Nx,Ny,Nt,Mx,Mt,Qx,Qy,Qt,x0,x1,y0,y1,t0,t1,mtype="psia"):
    
    models = []
    points = []
    Lx = x1 - x0
    Ly = y1 - y0
    tf = t1 - t0
    for kx in range(Nx):
        
        model_for_x = []
        point_for_x = []
        x_min = Lx/Nx * kx + x0
        x_max = Lx/Nx * (kx+1) + x0
        x_devide = np.linspace(x_min, x_max, Qx + 1)
        
        for ky in range(Ny):
            
            model_for_xy = []
            point_for_xy = []
            y_min = Ly/Nx * ky + y0
            y_max = Ly/Nx * (ky+1) + y0
            y_devide = np.linspace(y_min, y_max, Qy + 1)
            
            for n in range(Nt):
                t_min = tf/Nt * n + t0
                t_max = tf/Nt * (n+1) + t0
                if mtype == "psia":
                    model = local_mul_rep(space_features = 2, out_features = 1, hidden_layers = 1, Mx = Mx, Mt = Mt, x_min = x_min, 
                                        x_max = x_max, y_min = y_min, y_max = y_max, t_min = t_min, t_max = t_max)
                elif mtype == "psib":
                    model = local_mul_t_psib_rep(space_features = 2, out_features = 1, hidden_layers = 1, Mx = Mx, Mt = Mt, x_min = x_min, x_max = x_max, y_min = y_min, y_max = y_max, t_min = t_min, t_max = t_max, t0=t0, t1=t1)
                else:
                    raise NotImplementedError

                model = model.apply(weights_init)
                model = model.double()
                for param in model.parameters():
                    param.requires_grad = False
                
                model_for_xy.append(model)
                
                t_devide = np.linspace(t_min, t_max, Qt + 1)
                grid = np.array(list(itertools.product(x_devide,y_devide,t_devide))).reshape(Qx+1,Qy+1,Qt+1,3)
                point_for_xy.append(torch.tensor(grid,requires_grad=True))
            
            model_for_x.append(model_for_xy)
            point_for_x.append(point_for_xy)
        
        models.append(model_for_x)
        points.append(point_for_x)
    
    return(models,points)


def get_anal_u(vanal_u, points, Nx, Ny, Qx, Qy, nt=None, qt=None, tshift=0):

    if nt is None:
        nt = 0
    if qt is None:
        qt = 0

    point = list()
    
    for kx in range(Nx):
        point_x = list()
        for ky in range(Ny) :
            point_x.append(points[kx][ky][nt][:Qx, :Qy, qt, :].detach().numpy().reshape((Qx, Qy, 3)))
        point_x = np.concatenate(point_x, axis=1)
        point.append(point_x)
    
    point = np.concatenate(point, axis=0)
    
    u_value = vanal_u(point[:, :, 0], point[:, :, 1], point[:, :, 2] + tshift).reshape((Nx*Qx, Nx*Qy, 1))

    return u_value


def get_num_u(models, points, w, Nx, Ny, Nt, M, Qx, Qy, Qt, nt=None, qt=None):
    
    if nt is None:
        nt = Nt - 1
    if qt is None:
        qt = Qt
    
    u_value = list()

    for kx in range(Nx):
        u_value_x = list()
        for ky in range(Ny):
            model = models[kx][ky][nt]
            point = points[kx][ky][nt][:Qx, :Qy, qt, :]
            base_values = model(point).detach().numpy()
            M0 = (kx * Ny * Nt + ky * Nt + nt) * M
            
            u_value_x.append(np.dot(base_values, w[M0: M0 + M, :]))

        u_value_x = np.concatenate(u_value_x, axis=1)
        u_value.append(u_value_x)

    u_value = np.concatenate(u_value, axis=0)

    return u_value


def get_num_ut(models, points, w, Nx, Ny, Nt, M, Qx, Qy, Qt, nt=None, qt=None):

    if nt is None:
        nt = Nt - 1
    if qt is None:
        qt = Qt
    
    ut_value = list()

    for kx in range(Nx):
        ut_value_x = list()
        for ky in range(Ny):
            model = models[kx][ky][nt]
            point = points[kx][ky][nt][:Qx, :Qy, qt, :]
            base_values = model(point)
            base_grads = list()
            for i in range(M):
                grad = torch.autograd.grad(outputs=base_values[:, :, i], inputs=point, \
                    grad_outputs=torch.ones_like(base_values[:, :, i]), \
                    create_graph = True, retain_graph = True)[0]
                base_grads.append(grad.detach().numpy())
            base_grads = np.array(base_grads).swapaxes(0, 3)
            base_t_grads = base_grads[2, :, :, :]
            M0 = (kx * Ny * Nt + ky * Ny + nt) * M
            
            ut_value_x.append(np.dot(base_t_grads, w[M0: M0 + M, :]))

        ut_value_x = np.concatenate(ut_value_x, axis=1)
        ut_value.append(ut_value_x)

    ut_value = np.concatenate(ut_value, axis=0)

    return ut_value


def get_num_u_tpsib(models, points, w, Nx, Nt, M, Qx, Qt, nt=None, qt=None):
    
    if nt is None:
        nt = Nt - 1
    if qt is None:
        qt = Qt
    
    u_value = list()

    for k in range(Nx):

        if nt > 0:
            n_0 = nt - 1
        else:
            n_0 = nt
        if nt < Nt - 1:
            n_1 = nt + 2
        else:
            n_1 = nt + 1

        point = points[k][nt][:Qx, qt, :]
        u_value_ = np.zeros((Qx, 1))

        for n_s in range(n_0, n_1):
            M0 = (k * Nt + n_s) * M
            model = models[k][n_s]
            base_values = model(point).detach().numpy()
            u_value_ += np.dot(base_values, w[M0: M0 + M, :])

        u_value.append(u_value_)

    u_value = np.concatenate(u_value, axis=0)

    return u_value


def get_num_ut_tpsib(models, points, w, Nx, Nt, M, Qx, Qt, nt=None, qt=None):

    if nt is None:
        nt = Nt - 1
    if qt is None:
        qt = Qt
    
    ut_value = list()

    for k in range(Nx):
        
        if nt > 0:
            n_0 = nt - 1
        else:
            n_0 = nt
        if nt < Nt - 1:
            n_1 = nt + 2
        else:
            n_1 = nt + 1
        
        point = points[k][nt][:Qx, qt, :]
        ut_value_ = np.zeros((Qx, 1))

        for n_s in range(n_0, n_1):
            M0 = (k * Nt + n_s) * M
            model = models[k][n_s]
            base_values = model(point)
            base_grads = list()
            for i in range(M):
                grad = torch.autograd.grad(outputs=base_values[:, i], inputs=point, \
                    grad_outputs=torch.ones_like(base_values[:,i]), \
                    create_graph = True, retain_graph = True)[0]
                base_grads.append(grad.detach().numpy())
            base_grads = np.array(base_grads).swapaxes(0, 2)
            base_t_grads = base_grads[1, :, :]
            ut_value_ += np.dot(base_t_grads, w[M0: M0 + M, :])

        ut_value.append(ut_value_)

    ut_value = np.concatenate(ut_value, axis=0)

    return ut_value


def solve_lst_square(A, f, moore=False):

    # rescaling
    max_value = 10.0
    for i in range(len(A)):
        if np.abs(A[i,:]).max()==0:
            print("error line : ",i)
            continue
        ratio = max_value/np.abs(A[i,:]).max()
        A[i,:] = A[i,:]*ratio
        f[i] = f[i]*ratio
    
    # solve
    if moore:
        inv_coeff_mat = pinv(A)  # moore-penrose inverse, shape: (n_units,n_colloc+2)
        w = np.matmul(inv_coeff_mat, f)
    else:
        w = lstsq(A,f)[0]

    print(np.linalg.norm(np.matmul(A, w) - f), np.linalg.norm(np.matmul(A, w) - f) / np.linalg.norm(f))

    return w


def solve_lst_square_lowrank(A, f, moore=False):

    # rescaling
    max_value = 10.0
    for i in range(A.shape[0]):

        if np.abs(A[i,:]).max()==0:
            print("error line : ",i)
            continue
        ratio = max_value/np.abs(A[i,:]).max()
        A[i,:] = A[i,:]*ratio
        f[i] = f[i]*ratio
    
    # solve

    rs = list()
    rrs = list()

    w_ref = lstsq(A.toarray(), f)[0]
    r_ref = np.linalg.norm(A @ w_ref - f)
    rr_ref = np.linalg.norm(A @ w_ref - f) / np.linalg.norm(f)

    for i in range(10):

        q = A.shape[1] // 10 * (i + 1)

        print(A.shape, q)

        u, s, vh = svds(A=A, k=q-1)
        # u, s, vh = svd(a=A.toarray())

        quanzhong = ((u.T @ f) / s.reshape(-1, 1)) 

        w = vh.T @ quanzhong
        w = w.reshape(-1, 1)

        r = np.linalg.norm(A @ w - f)
        rr = np.linalg.norm(A @ w - f) / np.linalg.norm(f)

        rs.append(r)
        rrs.append(rr)

        fig, ax = plt.subplots()
        ax.plot(np.arange(q-1)+1, s[::-1], c="r")
        ax.set_yscale("log")
        ax.set_ylabel("Singularity Values", fontsize=16)
        ax.set_xlabel("Singularity Index", fontsize=16)

        ax2 = ax.twinx()
        ax2.plot(np.arange(q-1)+1, np.abs(quanzhong[::-1]), c="g")
        ax2.set_yscale("log")
        ax2.set_ylabel("Absolute Weights on Feature Vector", fontsize=16)

        ax.set_title(f"{q:d} Singularities", fontsize=20)

        fig.tight_layout()
        fig.savefig(f"singularities-{q:d}.png")
        plt.clf(); plt.cla(); plt.cla()

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

    ax[0].plot((np.arange(10) + 1) * A.shape[1] // 10, rs)
    ax[0].plot([A.shape[1] // 10, A.shape[1]], [r_ref, r_ref], label="reference")
    ax[0].set_title("Absolute Error")
    ax[0].set_xlabel("Number of Singularities Used")
    ax[0].set_yscale("log")
    ax[0].legend(loc="best")

    ax[1].plot((np.arange(10) + 1) * A.shape[1] // 10, rrs)
    ax[1].plot([A.shape[1] // 10, A.shape[1]], [rr_ref, rr_ref], label="reference")
    ax[1].set_title("Relative Error")
    ax[1].set_xlabel("Number of Singularities Used")
    ax[1].set_yscale("log")
    ax[1].legend(loc="best")

    fig.tight_layout()
    fig.savefig(f"sing.png")
    plt.clf(); plt.cla(); plt.cla()

    # u, s, vh = torch.svd_lowrank(A=torch.from_numpy(A).cuda(), q=q)
    # w = vh @ (u.T @ torch.from_numpy(f).cuda() / s.view(-1, 1))
    # w = w.detach().cpu().numpy()

    print(np.linalg.norm(A @ w - f), np.linalg.norm(A @ w - f) / np.linalg.norm(f))

    return w


def obtain_result(models,vanal_u=None,inter_indicator=None,Nx=1,Ny=1,Nt=1,M=1,Mx=None,Mt=None,Qx=1,Qy=1,Qt=1,w=None,x0=0,x1=1,y0=0,y1=1,t0=0,t1=1,test_Qx=None,test_Qy=None,test_Qt=None,block=0):

    Lx = x1 - x0
    Ly = y1 - y0
    tf = t1 - t0

    if Mx is not None and Mt is not None:
        M = Mx * Mt

    numerical_values = []
    true_values = []

    if test_Qx is None:
        test_Qx = 3 * Qx
    if test_Qy is None:
        test_Qy = 3 * Qy
    if test_Qt is None:
        test_Qt = 3 * Qt

    tshift = block * tf

    if inter_indicator is not None:

        is_inter = list()
    
    for kx in range(Nx):
        x_min = Lx/Nx * kx + x0
        x_max = Lx/Nx * (kx+1) + x0
        x_devide = np.linspace(x_min, x_max, test_Qx + 1)[:test_Qx]
        true_value_x = []
        numerical_value_x = []
        is_inter_x = []

        for ky in range(Ny):
            y_min = Ly/Ny * ky + y0
            y_max = Ly/Ny * (ky+1) + y0
            y_devide = np.linspace(y_min, y_max, test_Qy + 1)[:test_Qy]
            true_value_xy = []
            numerical_value_xy = []

            if inter_indicator is not None:
                grid_xy = np.array(list(itertools.product(x_devide,y_devide))).reshape(test_Qx,test_Qy,2)
                is_inter_xy = inter_indicator(grid_xy[:, :, 0], grid_xy[:, :, 1])

                is_inter_x.append(is_inter_xy)

            for n in range(Nt):
                # forward and grad
                t_min = tf/Nt * n
                t_max = tf/Nt * (n+1)
                t_devide = np.linspace(t_min, t_max, test_Qt + 1)[:test_Qt]
                
                grid = np.array(list(itertools.product(x_devide,y_devide,t_devide))).reshape(test_Qx,test_Qy,test_Qt,3)
                test_point = torch.tensor(grid,requires_grad=True)
                in_ = test_point.detach().numpy()

                if vanal_u is not None:
                    true_value = vanal_u(in_[:, :, :, 0], in_[:, :, :, 1], in_[:, :, :, 2] + tshift)
                    true_value_xy.append(true_value)
                
                values = models[kx][ky][n](test_point).detach().numpy()
                numerical_value = np.dot(values, w[(kx*Ny*Nt+ky*Nt+n)*M : (kx*Ny*Nt+ky*Nt+n+1)*M,:]).reshape(test_Qx,test_Qy,test_Qt)
                
                numerical_value_xy.append(numerical_value)

            if vanal_u is not None:
                true_value_xy = np.concatenate(true_value_xy, axis=2)
                true_value_x.append(true_value_xy)
            
            numerical_value_xy = np.concatenate(numerical_value_xy, axis=2)
            numerical_value_x.append(numerical_value_xy)
            
        if vanal_u is not None:
            true_value_x = np.concatenate(true_value_x, axis=1)
            true_values.append(true_value_x)
            
        numerical_value_x = np.concatenate(numerical_value_x, axis=1)
        numerical_values.append(numerical_value_x)

        if inter_indicator is not None:
            
            is_inter_x = np.concatenate(is_inter_x, axis=1)
            is_inter.append(is_inter_x)

    if vanal_u is not None:
        true_values = np.concatenate(true_values, axis=0)
    else:
        true_values = None

    numerical_values = np.concatenate(numerical_values, axis=0)

    if inter_indicator is not None:
        is_inter = np.concatenate(is_inter, axis=0)
    else:
        is_inter = None

    return true_values, numerical_values, is_inter


def test(models,inter_indicator=None,vanal_u=None,Nx=1,Ny=1,Nt=1,M=1,Mx=None,Mt=None,Qx=1,Qy=1,Qt=1,w=None,x0=0,x1=1,y0=0,y1=1,t0=0,t1=1,test_Qx=None,test_Qy=None,test_Qt=None,block=0):

    tf = t1 - t0

    if Mx is not None and Mt is not None:
        M = Mx * Mt

    tshift = block * tf

    true_values, numerical_values, is_inter = obtain_result(models=models, vanal_u=vanal_u, \
        Nx=Nx, Ny=Ny, Nt=Nt, M=M, Mx=Mx, Mt=Mt, Qx=Qx, Qy=Qy, Qt=Qt, \
        w=w, \
        x0=x0, x1=x1, y0=y0, y1=y1, t0=t0, t1=t1, \
        test_Qx=test_Qx, test_Qy=test_Qy, test_Qt=test_Qt, \
        block=block)

    if vanal_u is not None:

        epsilon = np.abs(true_values - numerical_values)

        epsilon_inter = epsilon[is_inter, :]

        e = epsilon_inter.reshape((-1,1))
        L_i = utils.L_inf_error(e)
        L_2 = utils.L_2_error(e)
        print('********************* ERROR *********************')
        print(f"Block={block:d},t0={tshift:.2f},t1={tshift + tf:.2f}")
        if Mx is not None and Mt is not None:
            print('Nx={:d},Ny={:d},Nt={:d},Mx={:d},Mt={:d},Qx={:d},Qy={:d},Qt={:d}'.format(Nx,Ny,Nt,Mx,Mt,Qx,Qy,Qt))
        else:
            print('Nx={:d},Ny={:d},Nt={:d},M={:d},Qx={:d},Qy={:d},Qt={:d}'.format(Nx,Ny,Nt,M,Qx,Qy,Qt))
        print('L_inf={:.2e}'.format(L_i),'L_2={:.2e}'.format(L_2))
        print("x-axis 边值条件误差")
        print("{:.2e} {:.2e}".format(np.max(epsilon[0,:,:]),np.max(epsilon[-1,:,:])))
        print("y-axis 边值条件误差")
        print("{:.2e} {:.2e}".format(np.max(epsilon[:,0,:]),np.max(epsilon[:,-1,:])))
        print("t-axis 初值、终值误差")
        print("{:.2e} {:.2e}".format(np.max(epsilon_inter[:,0]),np.max(epsilon_inter[:,-1])))
        # np.save('./epsilon_psi2.npy',epsilon)

    else:
        L_i = None
        L_2 = None

    return true_values, numerical_values, L_i, L_2


if __name__ == "__main__":
    model = local_mul_t_psib_rep(space_features=1, out_features=1, hidden_layers=1, Mx=10, Mt=1, x_min=0.0, x_max=1.0, t_min=0.0, t_max=1.0)
    x = torch.rand((20, 2))
    x[:, 1] -= -1
    out = model(x)

    print(out.min(), out.max())
