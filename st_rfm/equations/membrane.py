# -*- coding: utf-8 -*-

import numpy as np
from .geometry import Circle, Rectangular, Geometry_System

Pi = np.pi


class Membrane_CO_Geo(Geometry_System):

    def __init__(self, x0, y0, x1, y1, cxs, cys, crs, rx0s, ry0s, rx1s, ry1s) -> None:
        super().__init__()

        self.add_solid(Rectangular(x0=x0, y0=y0, x1=x1, y1=y1, name="solid-bound"))

        for i, (cx, cy, cr) in enumerate(zip(cxs, cys, crs)):

            self.add_obstacle(Circle(x0=cx, y0=cy, r=cr, name=f"obstacle-circle-{i:d}"))

        for i, (rx0, ry0, rx1, ry1) in enumerate(zip(rx0s, ry0s, rx1s, ry1s)):

            self.add_obstacle(Rectangular(x0=rx0, y0=ry0, x1=rx1, y1=ry1, name=f"obstacle-rect-{i:d}"))

class Membrane(object):

    def __init__(self) -> None:

        self.x0 = 0.0
        self.x1 = 5.0
        self.y0 = 0.0
        self.y1 = 4.0

        self.Lx = self.x1 - self.x0
        self.Ly = self.y1 - self.y0

        self.coef_A = 1.0

        self.name = "twod-membrane"

        self.vanal_u = np.vectorize(self.anal_u)
        self.vanal_phi = np.vectorize(self.anal_phi)
        self.vanal_psi = np.vectorize(self.anal_psi)
        self.vanal_f = np.vectorize(self.anal_f)

        self.geometry = Membrane_CO_Geo(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1, cxs=[], cys=[], crs=[], rx0s=[], ry0s=[], rx1s=[], ry1s=[])

    def indicator(self, x, y):

        return self.geometry.indicator(x, y)


    def anal_u(self, x, y, t):

        mu = 3 * Pi / self.Lx
        nu = 3 * Pi / self.Ly
        lam = np.sqrt(np.power(mu, 2.0) + np.power(nu, 2.0))
        lam *= self.coef_A
        u11 = np.sin(mu * x) * np.sin(nu * y) * (2.0 * np.cos(lam * t) + 1.0 * np.sin(lam * t))

        mu = Pi / self.Lx
        nu = 2 * Pi / self.Ly
        lam = np.sqrt(np.power(mu, 2.0) + np.power(nu, 2.0))
        lam *= self.coef_A
        u12 = np.sin(mu * x) * np.sin(nu * y) * (1.0 * np.cos(lam * t) + 2.0 * np.sin(lam * t))

        u = u11 + u12
        # u = u11

        return u

    def anal_phi(self, x, y, t=0):

        mu = 3 * Pi / self.Lx
        nu = 3 * Pi / self.Ly
        phi11 = np.sin(mu * x) * np.sin(nu * y) * 2.0 * (np.power(x - (self.x0 + self.x1) / 2, 2.0) + np.power(y - (self.y0 + self.y1) / 2, 2.0))

        mu = Pi / self.Lx
        nu = 2 * Pi / self.Ly
        phi12 = np.sin(mu * x) * np.sin(nu * y) * 1.0 * (np.power(x - (self.x0 + self.x1) / 2, 2.0) + np.power(y - (self.y0 + self.y1) / 2, 2.0))

        phi = phi11 + phi12

        # phi = phi11

        return phi

    def anal_psi(self, x, y, t=0):

        mu = 3 * Pi / self.Lx
        nu = 3 * Pi / self.Ly
        lam = np.sqrt(np.power(mu, 2.0) + np.power(nu, 2.0))
        lam *= self.coef_A
        psi11 = np.sin(mu * x) * np.sin(nu * y) * 1.0 * lam

        mu = Pi / self.Lx
        nu = 2 * Pi / self.Ly
        lam = np.sqrt(np.power(mu, 2.0) + np.power(nu, 2.0))
        lam *= self.coef_A
        psi12 = np.sin(mu * x) * np.sin(nu * y) * 2.0 * lam

        psi = psi11 + psi12
        # psi = psi11

        return psi

    def anal_f(self, x, y, t):

        return 0


class Membrane_CO(object):

    def __init__(self) -> None:

        self.x0 = 0.0
        self.x1 = 5.0
        self.y0 = 0.0
        self.y1 = 4.0

        self.Lx = self.x1 - self.x0
        self.Ly = self.y1 - self.y0

        self.cxs = [0.9, 2.1, 2.8, 4.3, 1.1, 2.3, 3.3, 4.0, ]
        self.cys = [1.2, 2.0, 0.9, 1.1, 2.6, 3.0, 2.2, 3.0, ]
        self.crs = [0.5, 0.4, 0.55, 0.45, 0.4, 0.45, 0.3, 0.4, ]

        # self.cxs = [2.5, ]
        # self.cys = [2.5, ]
        # self.crs = [1.0, ]

        # self.rx0s = [self.x0 + self.Lx / 3,]
        # self.ry0s = [self.y0 + self.Ly / 3,]
        # self.rx1s = [self.x0 + 2 * self.Lx / 3,]
        # self.ry1s = [self.y0 + 2 * self.Ly / 3,]

        # self.cxs = []
        # self.cys = []
        # self.crs = []

        self.rx0s = []
        self.ry0s = []
        self.rx1s = []
        self.ry1s = []

        self.n_obstacles = len(self.cxs) + len(self.rx0s)

        self.geometry = Membrane_CO_Geo(x0=self.x0, y0=self.y0, x1=self.x1, y1=self.y1, cxs=self.cxs, cys=self.cys, crs=self.crs, rx0s=self.rx0s, ry0s=self.ry0s, rx1s=self.rx1s, ry1s=self.ry1s)

        self.coef_A = 1.0

        self.name = "test-twod-membrane-withobs"

        self.vanal_u = np.vectorize(self.anal_u)

        # initial conditions
        self.vanal_I_t0 = np.vectorize(self.anal_phi)
        self.vanal_I_t1 = np.vectorize(self.anal_psi)

        self.vanal_dirichlet_bc = np.vectorize(self.anal_dirichlet_bc)

        # partial differential equation conditions
        self.vanal_f = np.vectorize(self.anal_f)

    
    def indicator(self, x, y):

        return self.geometry.indicator(x, y)


    def anal_u(self, x, y, t):

        mu = 2 * Pi / self.Lx
        nu = 2 * Pi / self.Ly
        lam = np.sqrt(np.power(mu, 2.0) + np.power(nu, 2.0))
        lam *= self.coef_A
        u11 = np.sin(mu * x) * np.sin(nu * y) * (2.0 * np.cos(lam * t) + 1.0 * np.sin(lam * t))

        mu = Pi / self.Lx
        nu = 2 * Pi / self.Ly
        lam = np.sqrt(np.power(mu, 2.0) + np.power(nu, 2.0))
        lam *= self.coef_A
        u12 = np.sin(mu * x) * np.sin(nu * y) * (1.0 * np.cos(lam * t) + 2.0 * np.sin(lam * t))

        u = u11 + u12

        return u

    def anal_dirichlet_bc(self, x, y, t):

        return self.anal_u(x, y, t)


    def anal_phi(self, x, y, t=0):

        mu = 2 * Pi / self.Lx
        nu = 2 * Pi / self.Ly
        phi11 = np.sin(mu * x) * np.sin(nu * y) * 2.0 * (np.power(x - (self.x0 + self.x1) / 2, 2.0) + np.power(y - (self.y0 + self.y1) / 2, 2.0))

        mu = Pi / self.Lx
        nu = 2 * Pi / self.Ly
        phi12 = np.sin(mu * x) * np.sin(nu * y) * 1.0# * (np.power(x - (self.x0 + self.x1) / 2, 2.0) + np.power(y - (self.y0 + self.y1) / 2, 2.0))

        phi = phi11 + phi12

        return phi

    def anal_psi(self, x, y, t=0):

        mu = 2 * Pi / self.Lx
        nu = 2 * Pi / self.Ly
        lam = np.sqrt(np.power(mu, 2.0) + np.power(nu, 2.0))
        lam *= self.coef_A
        psi11 = np.sin(mu * x) * np.sin(nu * y) * 1.0 * lam

        mu = Pi / self.Lx
        nu = 2 * Pi / self.Ly
        lam = np.sqrt(np.power(mu, 2.0) + np.power(nu, 2.0))
        lam *= self.coef_A
        psi12 = np.sin(mu * x) * np.sin(nu * y) * 2.0 * lam

        psi = psi11 + psi12
        # psi = psi11

        return psi

    def anal_f(self, x, y, t):

        # f = np.sin(t)
        f = 1

        return f

if __name__ == "__main__":

    import matplotlib as mpl
    import matplotlib.pyplot as plt

    x = np.arange(0.0, 5.0, step=0.01)
    y = np.arange(0.0, 4.0, step=0.01)

    xy = np.meshgrid(x, y)
    xy = np.stack(xy, axis=2)

    eqn = Membrane_CO()

    sign = eqn.indicator(xy[:, :, 0], xy[:, :, 1])

    fig, ax = plt.subplots()

    cmap = mpl.colors.ListedColormap(colors=["gray", "wheat"], name="mybin")
    im = ax.imshow(sign.astype(np.float64), cmap=cmap)
    ax.set_axis_off()
    # fig.colorbar(im, ax=ax)

    fig.savefig("membrane_wo.png")

    plt.clf(); plt.cla(); plt.close()
