import math
import numpy as np

# Creating a 2D Vector with x and y components
# __init__ method to initialize them, if not default is 0
class Vec2f:
    def __init__(self, x=0, y=0):
        self.v2 = np.array([x, y])
        # self.x = x
        # self.y = y


# Creating a 3D Vector with x, y, and z components
class Vec3f:
    def __init__(self, x=0, y=0, z=0):
        self.v3 = np.array([x, y, z])
        # self.x = x
        # self.y = y
        # self.z = z


# Method that allows to access the components of the vectors with vector[i] notation
    def __getitem__(self, index):
        if index == 0:
            # return self.x
            return self.v3[0]
        elif index == 1:
            # return self.y
            return self.v3[1]
        elif index == 2:
            # return self.z
            return self.v3[2]
        else:
            raise IndexError("Vec3f index out of range")

    def __sub__(self, other):
        # return Vec3f(self.x - other.x, self.y - other.y, self.z - other.z)
        return Vec3f(self.v3[0]-other.v3[0], self.v3[1]-other.v3[1], self.v3[2]-other.v3[2])

    def dot(self, other):
        # return self.x * other.x + self.y * other.y + self.z * other.z
        return self.v3[0] * other.v3[0] + self.v3[1] * other.v3[1] + self.v3[2] * other.v3[2]

    def length(self):
        # return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)
        return math.sqrt(self.v3[0] * self.v3[0] + self.v3[1] * self.v3[1] + self.v3[2] * self.v3[2])

    def normalize(self):
        # length = math.sqrt(self.x ** 2 + self.y ** 2 + self.z ** 2)
        length = math.sqrt(self.v3[0] ** 2 + self.v3[1] ** 2 + self.v3[2] ** 2)
        if length != 0:
            return Vec3f(self.v3[0] / length, self.v3[1] / length, self.v3[2] / length)
            # return Vec3f(self.x / length, self.y / length, self.z / length)
        else:
            return Vec3f()

    def cross(self, other):
        if isinstance(other, Vec3f):
            return Vec3f(
                self.v3[1] * other.v3[2] - self.v3[2] * other.v3[1],
                self.v3[2] * other.v3[0] - self.v3[0] * other.v3[2],
                self.v3[0] * other.v3[1] - self.v3[1] * other.v3[0]
            )
            # return Vec3f(
            #     self.y * other.z - self.z * other.y,
            #     self.z * other.x - self.x * other.z,
            #     self.x * other.y - self.y * other.x
            # )

    def norm(self):
        return math.sqrt(self.v3[0] * self.v3[0] + self.v3[1] * self.v3[1] + self.v3[2] * self.v3[2])
        # return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)


    def __mul__(self, other):
        if isinstance(other, Vec3f):
            return Vec3f(self.v3[0] * other.v3[0], self.v3[1] * other.v3[1], self.v3[2] * other.v3[2])
            # return Vec3f(self.x * other.x, self.y * other.y, self.z * other.z)
        elif isinstance(other, (int, float)):
            return Vec3f(self.v3[0] * other, self.v3[1] * other, self.v3[2] * other)
            # return Vec3f(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __add__(self, other):
        return Vec3f(self.v3[0] + other.v3[0], self.v3[1] + other.v3[1], self.v3[2] + other.v3[2])
        # return Vec3f(self.x + other.x, self.y + other.y, self.z + other.z)

    def __neg__(self):
        return Vec3f(-self.v3[0], -self.v3[1], -self.v3[2])
        # return Vec3f(-self.x, -self.y, -self.z)

    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Vec3f(self.v3[0] / other, self.v3[1] / other, self.v3[2] / other)
            # return Vec3f(self.x / other, self.y / other, self.z / other)
        elif isinstance(other, Vec3f):
            return Vec3f(self.v3[0] / other.v3[0], self.v3[1] / other.v3[1], self.v3[2] / other.v3[2])
            # return Vec3f(self.x / other.x, self.y / other.y, self.z / other.z)
        else:
            raise TypeError("Unsupported operand type for division.")


# Creating a 3D Vector with x, y, and z components.
# It has int components, instead of float
class Vec3i:
    def __init__(self, x=0, y=0, z=0):
        self.vi = np.array([x, y, z])
        # self.x = x
        # self.y = y
        # self.z = z

    def __getitem__(self, index):
        if index == 0:
            # return self.x
            return self.vi[0]
        elif index == 1:
            # return self.y
            return self.vi[1]
        elif index == 2:
            # return self.z
            return self.vi[2]
        else:
            raise IndexError("Vec3f index out of range")


# Creating a 4D Vector with x, y, z, and w components.
class Vec4f:
    def __init__(self, x=0, y=0, z=0, w=0):
        self.v4 = np.array([x, y, z, w])
        # self.x = x
        # self.y = y
        # self.z = z
        # self.w = w

    # Method that allows to access the components of the vectors with vector[i] notation
    def __getitem__(self, index):
        if index == 0:
            return self.v4[0]
            # return self.x
        elif index == 1:
            return self.v4[1]
            # return self.y
        elif index == 2:
            return self.v4[2]
            # return self.z
        elif index == 3:
            return self.v4[3]
            # return self.w
        else:
            raise IndexError("Vec3f index out of range")