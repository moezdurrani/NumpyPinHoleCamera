import math
from vectors import *

class Sphere:
    def __init__(self, center, radius, material):
        self.center = center
        self.radius = radius
        self.material = material

    def ray_intersect(self, orig, dir, t0):

        orig = np.array(orig)
        dir = np.array(dir)
        center = np.array(self.center)

        OtC = center - orig
        dt = np.dot(OtC, dir)
        d2 = np.dot(OtC, OtC) - dt * dt

        if d2 > self.radius ** 2:
            return False

        thc = np.sqrt(self.radius ** 2 - d2)
        t0[0] = dt - thc
        t1 = dt + thc

        if t0[0] < 0:
            t0[0] = t1

        if t0[0] < 0:
            return False

        return True
# orig, origin of the ray
# dir, direction of the ray
# OtC, vector from origin of the ray to the center of the sphere
# dtt, dot product of OtC and dir (the direction of the ray)
# d2, the squared length of OtC minus the squared length of the projection of OtC onto dir.
# d2 is the perpendicular distance between the ray and the center of the radius