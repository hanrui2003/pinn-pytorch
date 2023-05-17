import numpy as np
from math import cos, pi, sin, sqrt
import sys
import time


def F1vector(xstart, xend, ystart, yend):
    return qx_o[ystart:yend, xstart:xend]


def F2vector(xstart, xend, ystart, yend):
    return np.multiply(h_o[ystart:yend, xstart:xend], u_o[ystart:yend, xstart:xend] ** 2) + 0.5 * g * h_o[ystart:yend,
                                                                                                      xstart:xend] ** 2


def F3vector(xstart, xend, ystart, yend):
    return qxy_o[ystart:yend, xstart:xend]


def G1vector(xstart, xend, ystart, yend):
    return qy_o[ystart:yend, xstart:xend]


def G2vector(xstart, xend, ystart, yend):
    return qxy_o[ystart:yend, xstart:xend]


def G3vector(xstart, xend, ystart, yend):
    return np.multiply(h_o[ystart:yend, xstart:xend], v_o[ystart:yend, xstart:xend] ** 2) + 0.5 * g * h_o[ystart:yend,
                                                                                                      xstart:xend] ** 2


def BC():
    # x direction
    h_o[:, 0] = h_o[:, 1]
    h_o[:, -1] = h_o[:, -2]
    qx_o[:, 0] = -qx_o[:, 1]
    qx_o[:, -1] = -qx_o[:, -2]
    qy_o[:, 0] = qy_o[:, 1]
    qy_o[:, -1] = qy_o[:, -2]

    # y direction
    h_o[0, :] = h_o[1, :]
    h_o[-1, :] = h_o[-2, :]
    qx_o[0, :] = qx_o[1, :]
    qx_o[-1, :] = qx_o[-2, :]
    qy_o[0, :] = -qy_o[1, :]
    qy_o[-1, :] = -qy_o[-2, :]


def profile():
    # b[1:-1,1:-1]=1-xm
    # b[1:-1,1:int(2*nx/3)+1]=2/3-xm[:, :int(2*nx/3)]
    # b[1:-1,int(2*nx/3)+1:]=0

    # dam break
    # b[1:-1,1:-1]=0

    # slant profile
    # b[1:-1,1:int(2*nx/3)+1]=2/3-xm[:, :int(2*nx/3)]
    # b[1:-1,int(2*nx/3)+1:]=0

    # mountain profile
    # b[1:-1,1:-1] =1-np.sin(np.sqrt((2*(xm-0.5))**2 +0.1* (5*(ym-0.5))**2) )

    # #explorating data
    # b[:,0]=b[:,1]+(b[:,1]-b[:,2])
    # b[0,:]=b[1,:]+(b[1,:]-b[2,:])

    # new case

    b[1:-1, 1:-1] = -h0 * (1 - ((xm - l / 2) ** 2 + (ym - l / 2) ** 2) / a ** 2)


def h_init():
    # dam break
    # h_o[:, int(nx/2):]=0.5
    # h_o[:, :int(nx/2)]=1

    # slant profile
    # h_o[:, int(nx/2.5):-int(nx/2.5)]=0.3

    # moutain profile
    # h_o[int(nx/2.5):-int(nx/2.5), int(nx/4):int(nx/2.5)]=0.5

    # new case
    h_o[1:-1, 1:-1] = eta * h0 * (2 * (xm - l / 2) - eta) - b[1:-1, 1:-1]
    h_o[h_o < 0] = 0


# mesh code
l = 4.0
b = 4.0
nx = 171
ny = 171
runtime = 2
at_time = [0]

xgrid = np.linspace(0, l, nx, dtype=float)
ygrid = np.linspace(0, b, ny, dtype=float)
print(ygrid)
xm, ym = np.meshgrid(xgrid, ygrid)

dx = l / (nx - 1)
dy = b / (ny - 1)

# print("xgrid",xgrid)
cfl_limit = 0.2

g = 9.8

iniGRIDFunc = lambda m: [np.zeros((ny + 2, nx + 2), dtype=float) for _ in range(m)]
h_n, qx_n, qy_n, u_n, v_n = iniGRIDFunc(5)
h_o, qx_o, qy_o, qxy_o, u_o, v_o = iniGRIDFunc(6)

# print(u_o)
h = [h_o]
v = [v_o]
u = [u_o]

b = np.zeros((ny + 2, nx + 2), dtype=float)
lamdax = np.zeros((ny + 2, nx + 2), dtype=float)
lamday = np.zeros((ny + 2, nx + 2), dtype=float)
hstable = np.zeros((ny + 2, nx + 2), dtype=float)

# input conditions
# h[0,:int(nx/2),int(ny/4):-int(ny/4)]=1


# u[0,:,:]=0.2
# v_o[:,:]=eta*w
profile()
h_init()

qx_o[:, :] = np.multiply(h_o[:, :], u_o[:, :])
qy_o[:, :] = np.multiply(h_o[:, :], v_o[:, :])
qxy_o[:, :] = np.multiply(qx_o[:, :], v_o[:, :])

BC()

start2 = time.time()

t = 0

while at_time[t] < runtime:
    # print(t+1)
    # print(t+1)
    lamdax = max(np.max(np.abs(u_o[:, :] + np.sqrt(g * h_o[:, :]))), np.max(np.abs(u_o[:, :] - np.sqrt(g * h_o[:, :]))))
    lamday = max(np.max(np.abs(v_o[:, :] + np.sqrt(g * h_o[:, :]))), np.max(np.abs(v_o[:, :] - np.sqrt(g * h_o[:, :]))))
    # print(lamdax,lamday)

    maxlamda = max(lamday / dy, lamdax / dx)
    # print(lamdax, lamday)
    dt = cfl_limit / maxlamda
    # print(dt)
    # print(dt,at_time[t])

    t = t + 1
    hstable[1:-1, 1:-1] = (h_o[2:, 1:-1] + h_o[0:-2, 1:-1] + h_o[1:-1, 2:] + h_o[1:-1, 0:-2]) / 4.0

    h_n[1:-1, 1:-1] = (h_o[2:, 1:-1] + h_o[0:-2, 1:-1] + h_o[1:-1, 2:] + h_o[1:-1, 0:-2]) / 4.0 - dt * (
            F1vector(2, nx + 2, 1, -1) - F1vector(0, -2, 1, -1)) / (2.0 * dx) - dt * (
                              G1vector(1, -1, 2, ny + 2) - G1vector(1, -1, 0, -2)) / (2.0 * dy)

    qx_n[1:-1, 1:-1] = (qx_o[2:, 1:-1] + qx_o[0:-2, 1:-1] + qx_o[1:-1, 2:] + qx_o[1:-1, 0:-2]) / 4.0 - dt * (
            F2vector(2, nx + 2, 1, -1) - F2vector(0, -2, 1, -1)) / (2.0 * dx) - dt * (
                               G2vector(1, -1, 2, ny + 2) - G2vector(1, -1, 0, -2)) / (2.0 * dy) - dt * g * hstable[
                                                                                                            1:-1,
                                                                                                            1:-1] * (
                               b[1:-1, 1:-1] - b[1:-1, :-2]) / dx
    # $print(t*dt,qx[t,1:-1,1:-1])
    qy_n[1:-1, 1:-1] = (qy_o[2:, 1:-1] + qy_o[0:-2, 1:-1] + qy_o[1:-1, 2:] + qy_o[1:-1, 0:-2]) / 4.0 - dt * (
            F3vector(2, nx + 2, 1, -1) - F3vector(0, -2, 1, -1)) / (2.0 * dx) - dt * (
                               G3vector(1, -1, 2, ny + 2) - G3vector(1, -1, 0, -2)) / (2.0 * dy) - dt * g * hstable[
                                                                                                            1:-1,
                                                                                                            1:-1] * (
                               b[1:-1, 1:-1] - b[:-2, 1:-1]) / dy

    h_o = np.copy(h_n)
    qx_o = np.copy(qx_n)
    qy_o = np.copy(qy_n)

    BC()
    # print(h_o)
    # print()
    # print(qx_o)
    # print()
    # print(qy_o)

    u_o[:, :] = np.divide(qx_o[:, :], h_o[:, :], out=np.zeros_like(qx_o[:, :]), where=h_o[:, :] != 0)
    v_o[:, :] = np.divide(qy_o[:, :], h_o[:, :], out=np.zeros_like(qy_o[:, :]), where=h_o[:, :] != 0)
    # print(h_o)

    qxy_o[:, :] = np.multiply(qx_o[:, :], v_o[:, :])

    h.append(np.copy(h_o))
    u.append(np.copy(u_o))
    v.append(np.copy(v_o))
    # print("here",u[t])
    at_time.append(at_time[-1] + dt)
# if max( np.max(u[t,:,:])/dx, np.max(v[t,:,:])/dy )*dt >cfl_limit:
# 	print("CFL =",max( np.max(u[t,:,:])/dx, np.max(v[t,:,:]/dy)*dt ), "at u/v=", np.max(u[t,:,:]), np.max(v[t,:,:]) ," for dt and dx", dt,dx )
# 	sys.exit("CFL limit exeeded, either decrease dt or increase dx")


end2 = time.time()

print("code runtime", end2 - start2)
print("total iterations", t)

u = np.array(u)
h = np.array(h)
v = np.array(v)

# print(h.shape)

# print(u)
# print(h)
# print(u)
# print(b[:,:])
# print("inx",b[1:-1,1:-1]-b[1:-1,:-2])
# print("iny",b[1:-1,1:-1]-b[:-2,1:-1])
print("successfully computed")

# np.save('h',h)
# np.save('u',u)
# np.save('v',v)
# np.save('xgrid',xgrid)
# np.save('ygrid', ygrid)
# np.save('b',b)


# start1=time.time()
# for t in range(1, len(h[:,0])):

# 	for i in range(1,len(h[0,:-1])):
# 		# if t==21:
# 		# 	print(t,h[t,i])

# 		h[t,i]=(h[t-1,i+1] + h[t-1,i-1])/2 - lam*(F1(t-1,i+1) - F1(t-1,i-1))/2

# 		q[t,i]=(q[t-1,i+1] + q[t-1,i-1])/2 - lam*(F2(t-1,i+1) - F2(t-1,i-1))/2 - lam*g*h[t-1,i]*(b[i]-b[i-1])

# 	BC(t)

# 	for i in range(1,len(h[0,:-1])):
# 		if h[t,i]==0:
# 			u[t,i]=0
# 		else:
# 			u[t,i]=q[t,i]/h[t,i]


# end1=time.time()

# print("old code runtime", end1-start1)
