# -*- coding: utf-8 -*-

import numpy as np

Pi = np.pi

class Diffusion(object):

    def __init__(self) -> None:

        self.a1 = 0.0
        self.b1 = 5.0
        self.L = self.b1 - self.a1
        self.nu = 0.01
        self.tf = 1.0

        self.name = "oned-diff"


        self.vanal_u = np.vectorize(self.anal_u)
        self.vanal_f = np.vectorize(self.anal_f)

    def anal_u(self, x, t):

        u0 = 2 * np.cos(Pi * x + Pi / 5) + 3 / 2 * np.cos(2 * Pi * x - 3 * Pi / 5)
        u1 = 2 * np.cos(Pi * t + Pi / 5) + 3 / 2 * np.cos(2 * Pi * t - 3 * Pi / 5)
        
        u = u0 * u1

        return u

    def anal_f(self, x, t):

        u0 = 2 * np.cos(Pi * x + Pi / 5) + 3 / 2 * np.cos(2 * Pi * x - 3 * Pi / 5)
        u1 = 2 * np.cos(Pi * t + Pi / 5) + 3 / 2 * np.cos(2 * Pi * t - 3 * Pi / 5)

        u1t = -2 * Pi * np.sin(Pi * t + Pi / 5) - 3 * Pi * np.sin(2 * Pi * t - 3 * Pi / 5)
        ut = u0 * u1t

        u0xx = -2 * Pi * Pi * np.cos(Pi * x + Pi / 5) - 6 * Pi * Pi * np.cos(2 * Pi * x - 3 * Pi / 5)
        uxx = u0xx * u1
        
        f = ut - self.nu * uxx

        return f