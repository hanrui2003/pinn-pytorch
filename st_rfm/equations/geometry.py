# -*- coding: utf-8 -*-

import numpy as np

Pi = np.pi

class Geometry(object):

    def __init__(self, name="Geometry") -> None:
        self.name = name
        self.type = "Geom"

    def indicator(self, x, y):
        raise NotImplementedError

    def sampler(self,):
        raise NotImplementedError

    def dirichlet(self, x, y):
        raise NotImplementedError

    def neumann(self, x, y):
        raise NotImplementedError


class Circle(Geometry):

    def __init__(self, x0, y0, r, name="circle") -> None:
        super().__init__(name=name)

        self.type = "Circle"

        self.x0 = x0
        self.y0 = y0
        self.r = r 

    def indicator(self, x, y):

        f = - (np.power(x - self.x0, 2.0) + np.power(y - self.y0, 2.0)) + np.power(self.r, 2.0)

        return (f > 0)

    def sampler(self, n):
        
        angles = np.linspace(start=-Pi, stop=Pi, num=n, endpoint=False)

        x = np.cos(angles) * self.r + self.x0
        y = np.sin(angles) * self.r + self.y0

        data = np.stack([x, y], axis=1)

        return data


class Rectangular(Geometry):

    def __init__(self, x0, y0, x1, y1, name="rect") -> None:
        super().__init__(name=name)

        self.type = "Rect"

        self.x0 = x0
        self.y0 = y0
        self.x1 = x1
        self.y1 = y1

        self.Lx = x1 - x0
        self.Ly = y1 - y0

    def indicator(self, x, y):

        sx = (x >= self.x0) & (x < self.x1)
        sy = (y >= self.y0) & (y < self.y1)

        return np.bitwise_and(sx, sy)

    def sampler(self, n):

        l = np.linspace(start=0, stop=2 * (self.Lx + self.Ly), num=n, endpoint=False)

        i1 = np.where(l < self.Lx)[0]

        i2 = np.where((l >= self.Lx) & (l < (self.Lx + self.Ly)))[0]
        l[i2] -= self.Lx

        i3 = np.where((l >= (self.Lx + self.Ly)) & (l < (2 * self.Lx + self.Ly)))[0]
        l[i3] -= (self.Lx + self.Ly)

        i4 = np.where((l >= (2 * self.Lx + self.Ly)) & (l < (2 * self.Lx + 2 * self.Ly)))[0]
        l[i4] -= (2 * self.Lx + self.Ly)

        l1 = np.array([[self.x0 + _l, self.y0] for _l in l[i1]])
        l2 = np.array([[self.x1, self.y0 + _l] for _l in l[i2]])
        l3 = np.array([[self.x1 - _l, self.y1] for _l in l[i3]])
        l4 = np.array([[self.x0, self.y1 - _l] for _l in l[i4]])

        data = np.concatenate([l1, l2, l3, l4], axis=0)

        return data

class Geometry_System(object):

    def __init__(self) -> None:
        
        self.solids = list()
        self.obstacles = list()

    def add_solid(self, geom):

        self.solids.append(geom)

    def add_obstacle(self, geom):

        self.obstacles.append(geom)

    def indicator(self, x, y):

        solid_sign = False

        for geom in self.solids:

            solid_sign = np.bitwise_or(solid_sign, geom.indicator(x, y))

        obstacle_sign = False

        for geom in self.obstacles:

            obstacle_sign = np.bitwise_or(obstacle_sign, geom.indicator(x, y))

        sign = np.bitwise_and(solid_sign, np.bitwise_not(obstacle_sign))

        return sign
