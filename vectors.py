import math
import numpy as np

# Creating a 2D Vector with x and y components
# __init__ method to initialize them, if not default is 0
class Vec2f:
    def __init__(self, x=0, y=0):
        self.v2 = np.array([x, y])

# Creating a 3D Vector with x, y, and z components
class Vec3f:
    def __init__(self, x=0, y=0, z=0):
        self.v3 = np.array([x, y, z])

# Method that allows to access the components of the vectors with vector[i] notation
    def __getitem__(self, index):
        if index == 0:
            return self.v3[0]
        elif index == 1:
            return self.v3[1]
        elif index == 2:
            return self.v3[2]
        else:
            raise IndexError("Vec3f index out of range")

    # Subtract the other vector from current vector
    def __sub__(self, other):
        return Vec3f(*(self.v3 - other.v3))
        # return Vec3f(self.v3[0]-other.v3[0], self.v3[1]-other.v3[1], self.v3[2]-other.v3[2])

    def dot(self, other):
        return np.dot(self.v3, other.v3)
        # return self.v3[0] * other.v3[0] + self.v3[1] * other.v3[1] + self.v3[2] * other.v3[2]

    def length(self):
        return np.linalg.norm(self.v3)
        # return math.sqrt(self.v3[0] * self.v3[0] + self.v3[1] * self.v3[1] + self.v3[2] * self.v3[2])

    # Duplicate functions, length and norm, remove one of them
    def norm(self):
        return np.linalg.norm(self.v3)
        # return math.sqrt(self.v3[0] * self.v3[0] + self.v3[1] * self.v3[1] + self.v3[2] * self.v3[2])

    def normalize(self):
        length = np.linalg.norm(self.v3)
        # length = math.sqrt(self.v3[0] ** 2 + self.v3[1] ** 2 + self.v3[2] ** 2)
        if length != 0:
            return Vec3f(*(self.v3 / length))
            # return Vec3f(self.v3[0] / length, self.v3[1] / length, self.v3[2] / length)
        else:
            return Vec3f()

    def cross(self, other):
        if isinstance(other, Vec3f):
            return Vec3f(*np.cross(self.v3, other.v3))
            # return Vec3f(
            #     self.v3[1] * other.v3[2] - self.v3[2] * other.v3[1],
            #     self.v3[2] * other.v3[0] - self.v3[0] * other.v3[2],
            #     self.v3[0] * other.v3[1] - self.v3[1] * other.v3[0]
            # )

    def __mul__(self, other):
        if isinstance(other, Vec3f):
            return Vec3f(*(self.v3 * other.v3))
            # return Vec3f(self.v3[0] * other.v3[0], self.v3[1] * other.v3[1], self.v3[2] * other.v3[2])
        elif isinstance(other, (int, float)):
            return Vec3f(*(self.v3 * other))
            # return Vec3f(self.v3[0] * other, self.v3[1] * other, self.v3[2] * other)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __add__(self, other):
        return Vec3f(*(self.v3 + other.v3))
        # return Vec3f(self.v3[0] + other.v3[0], self.v3[1] + other.v3[1], self.v3[2] + other.v3[2])

    def __neg__(self):
        return Vec3f(*(-1 * self.v3))
        # return Vec3f(-self.v3[0], -self.v3[1], -self.v3[2])

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vec3f(*(self.v3 / other))
            # return Vec3f(self.v3[0] / other, self.v3[1] / other, self.v3[2] / other)
        elif isinstance(other, Vec3f):
            return Vec3f(*(self.v3 / other.v3))
            # return Vec3f(self.v3[0] / other.v3[0], self.v3[1] / other.v3[1], self.v3[2] / other.v3[2])
        else:
            raise TypeError("Unsupported operand type for division.")


# Creating a 3D Vector with x, y, and z components.
# It has int components, instead of float
class Vec3i:
    def __init__(self, x=0, y=0, z=0):
        self.vi = np.array([x, y, z])

    def __getitem__(self, index):
        if index == 0:
            return self.vi[0]
        elif index == 1:
            return self.vi[1]
        elif index == 2:
            return self.vi[2]
        else:
            raise IndexError("Vec3f index out of range")


# Creating a 4D Vector with x, y, z, and w components.
class Vec4f:
    def __init__(self, x=0, y=0, z=0, w=0):
        self.v4 = np.array([x, y, z, w])

    # Method that allows to access the components of the vectors with vector[i] notation
    def __getitem__(self, index):
        if index == 0:
            return self.v4[0]
        elif index == 1:
            return self.v4[1]
        elif index == 2:
            return self.v4[2]
        elif index == 3:
            return self.v4[3]
        else:
            raise IndexError("Vec3f index out of range")