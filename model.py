# import sys
# import numpy as np
# import pyrr
#
# class BVHNode:
#     def __init__(self, bbox, faces):
#         self.bbox = bbox
#         self.faces = faces
#         self.children = []
#
#     def intersect(self, ray_origin, ray_direction, tnear, tfar):
#         if not self.bbox.intersect_ray(ray_origin, ray_direction, tnear, tfar):
#             return False, tfar
#
#         if not self.children:
#             for face_index in self.faces:
#                 if model.ray_triangle_intersect(face_index, ray_origin, ray_direction, tnear, tfar):
#                     tfar = model.ray_triangle_intersect(face_index, ray_origin, ray_direction, tnear, tfar)
#             return True, tfar
#
#         hit, tfar = False, tfar
#         for child in self.children:
#             hit_child, tfar = child.intersect(ray_origin, ray_direction, tnear, tfar)
#             hit = hit or hit_child
#
#         return hit, tfar
#
# class Model:
#     def __init__(self, filename, material, x, y, z):
#         self.verts = []
#         self.faces = []
#         self.material = material
#         self.load_model(filename, x, y, z)
#         self.build_bvh()
#
#     def load_model(self, filename: str, x, y, z) -> None:
#         try:
#             with open(filename, 'r') as file:
#                 for line in file:
#                     line = line.strip()
#                     if line.startswith('v '):
#                         coordinates = line[2:].split()
#                         vertex = np.array([float(coordinates[0]) + x, float(coordinates[1]) + y, float(coordinates[2]) + z])
#                         self.verts.append(vertex)
#                     elif line.startswith('f '):
#                         indices = line[2:].split()
#                         face = np.array([int(indices[0]) - 1, int(indices[1]) - 1, int(indices[2]) - 1])
#                         self.faces.append(face)
#         except FileNotFoundError:
#             print(f"Failed to open {filename}")
#             sys.exit(1)
#
#         print(f"Vertices: {len(self.verts)} Faces: {len(self.faces)}")
#
#
#
#     def build_bvh(self):
#         bbox = pyrr.aabbc.create_from_points(np.array([v for v in self.verts], dtype=np.float32))
#         self.bvh_root = self.build_bvh_recursive(bbox, list(range(self.nfaces())))
#
#     def build_bvh_recursive(self, bbox, faces):
#         node = BVHNode(bbox, faces)
#
#         if len(faces) <= 4:
#             return node
#
#         centroids = [self.compute_face_centroid(self.faces[face_index]) for face_index in faces]
#         split_axis = bbox.maximum_extent()
#         centroids.sort(key=lambda c: c[split_axis])
#
#         mid = len(centroids) // 2
#         left_faces = [faces[i] for i in range(mid)]
#         right_faces = [faces[i] for i in range(mid, len(faces))]
#
#         left_bbox = pyrr.aabbc.create_from_points(np.array([self.compute_face_centroid(self.faces[face_index]) for face_index in left_faces], dtype=np.float32))
#         right_bbox = pyrr.aabbc.create_from_points(np.array([self.compute_face_centroid(self.faces[face_index]) for face_index in right_faces], dtype=np.float32))
#
#         node.children.append(self.build_bvh_recursive(left_bbox, left_faces))
#         node.children.append(self.build_bvh_recursive(right_bbox, right_faces))
#
#         return node
#
#     def compute_face_centroid(self, face):
#         v0 = self.verts[face[0]]
#         v1 = self.verts[face[1]]
#         v2 = self.verts[face[2]]
#         return (v0 + v1 + v2) / 3.0
#
#     def ray_triangle_intersect(self, fi: int, orig: np.ndarray, dir: np.ndarray, tnear: float, tfar=float('inf')) -> bool:
#
#         face_indices = self.faces[fi]
#
#         # Get vertices for the current face
#         v0 = self.verts[self.faces[fi][0]]
#         v1 = self.verts[self.faces[fi][1]]
#         v2 = self.verts[self.faces[fi][2]]
#
#         edge1 = v1 - v0
#         edge2 = v2 - v0
#
#         # Begin calculating determinant - also used to calculate u parameter
#         pvec = np.cross(dir, edge2)
#         det = np.dot(edge1, pvec)
#
#         if det < 1e-5:
#             return False
#
#         inv_det = 1.0 / det
#
#         # Calculate distance from v0 to ray origin
#         tvec = orig - v0
#
#         # Calculate u parameter and test bound
#         u = np.dot(tvec, pvec) * inv_det
#         if u < 0.0 or u > 1.0:
#             return False
#
#         # Prepare to test v parameter
#         qvec = np.cross(tvec, edge1)
#
#         v = np.dot(dir, qvec)
#         if v < 0 or u + v > det:
#             return False
#
#         # Calculate V parameter and test bound
#         v = np.dot(dir, qvec) * inv_det
#         if v < 0.0 or u + v > 1.0:
#             return False
#
#         # Calculate t, ray intersects triangle
#         t = np.dot(edge2, qvec) * inv_det
#
#         return (t > 1e-5) and (t < tfar)
#     def intersect_ray(self, ray_origin, ray_direction, tnear=0.0, tfar=float('inf')):
#         return self.bvh_root.intersect(ray_origin, ray_direction, tnear, tfar)
#
#     # ... (rest of the existing Model class methods)
#     def nverts(self) -> int:
#         return len(self.verts)
#
#     def nfaces(self) -> int:
#         return len(self.faces)
#
#     def get_face(self, index: int) -> np.ndarray:
#         assert 0 <= index < self.nfaces()
#         return self.faces[index]
#
#     def get_bbox(self) -> (np.ndarray, np.ndarray):
#         min_vertex = max_vertex = self.verts[0]
#         for vertex in self.verts[1:]:
#             for i in range(3):
#                 min_vertex = np.array([min(min_vertex[0], vertex[0]), min(min_vertex[1], vertex[1]),
#                                    min(min_vertex[2], vertex[2])])
#                 max_vertex = np.array([max(max_vertex[0], vertex[0]), max(max_vertex[1], vertex[1]),
#                                    max(max_vertex[2], vertex[2])])
#         return min_vertex, max_vertex
#
#     def point(self, i: int) -> np.ndarray:
#         assert 0 <= i < self.nverts()
#         return self.verts[i]
#     # def point(self, i: int) -> Vec3f:
#     #     assert 0 <= i < self.nverts()
#     #     return self.verts[i]
#
#     def vert(self, fi: int, li: int) -> int:
#         assert 0 <= fi < self.nfaces() and 0 <= li < 3
#         face = self.faces[fi]
#         if li == 0:
#             return face[0]
#         elif li == 1:
#             return face[1]
#         elif li == 2:
#             return face[2]
#
#     def __str__(self) -> str:
#         output = ""
#         for i in range(self.nverts()):
#             output += f"v {self.point(i)}\n"
#         for i in range(self.nfaces()):
#             output += "f "
#             for k in range(3):
#                 output += f"{self.vert(i, k) + 1} "
#             output += "\n"
#         return output
#
#     def compute_normal(self, face):
#         v0 = self.point(face[0])
#         v1 = self.point(face[1])
#         v2 = self.point(face[2])
#         edge1 = v1 - v0
#         edge2 = v2 - v0
#
#         N = np.cross(edge1, edge2)
#         Nmagnitude = np.linalg.norm(N)
#         N = N / Nmagnitude if Nmagnitude != 0 else N
#
#         return N
#
#     def rotate_x(self, angle_degrees):
#         center = np.array([0.0, 0.0, 0.0])
#         for vertex in self.verts:
#             center += np.array(vertex)
#         center /= len(self.verts)
#
#         # Convert the angle from degrees to radians
#         angle_radians = np.radians(angle_degrees)
#
#         # Create the rotation matrix for X-axis
#         rotation_matrix = np.array([
#             [1, 0, 0],
#             [0, np.cos(angle_radians), -np.sin(angle_radians)],
#             [0, np.sin(angle_radians), np.cos(angle_radians)]
#         ])
#
#         # Apply the rotation matrix to all vertices relative to the center
#         for i in range(self.nverts()):
#             # Translate the vertex to the origin
#             translated_vertex = self.verts[i] - center
#
#             # Convert the translated vertex to a NumPy array
#             vertex_array = np.array([translated_vertex[0], translated_vertex[1], translated_vertex[2]])
#
#             # Apply the rotation
#             rotated_vertex_array = np.dot(rotation_matrix, vertex_array)
#
#             # Update the vertex in the verts list
#             self.verts[i] = np.array([rotated_vertex_array[0], rotated_vertex_array[1], rotated_vertex_array[2]])
#
#         # Translate the model back to its original position
#         for i in range(self.nverts()):
#             self.verts[i] += center
#
#     def rotate_y(self, angle_degrees):
#         center = np.array([0.0, 0.0, 0.0])
#         for vertex in self.verts:
#             center += np.array(vertex)
#         center /= len(self.verts)
#
#         # Convert the angle from degrees to radians
#         angle_radians = np.radians(angle_degrees)
#
#         # Create the rotation matrix for Y-axis
#         rotation_matrix = np.array([
#             [np.cos(angle_radians), 0, np.sin(angle_radians)],
#             [0, 1, 0],
#             [-np.sin(angle_radians), 0, np.cos(angle_radians)]
#         ])
#
#         # Apply the rotation matrix to all vertices relative to the center
#         for i in range(self.nverts()):
#             # Translate the vertex to the origin
#             translated_vertex = self.verts[i] - center
#
#             # Convert the translated vertex to a NumPy array
#             vertex_array = np.array([translated_vertex[0], translated_vertex[1], translated_vertex[2]])
#
#             # Apply the rotation
#             rotated_vertex_array = np.dot(rotation_matrix, vertex_array)
#
#             # Update the vertex in the verts list
#             self.verts[i] = np.array([rotated_vertex_array[0], rotated_vertex_array[1], rotated_vertex_array[2]])
#
#         # Translate the model back to its original position
#         for i in range(self.nverts()):
#             self.verts[i] += center
#
#     def rotate_z(self, angle_degrees):
#         center = np.array([0.0, 0.0, 0.0])
#         for vertex in self.verts:
#             center += np.array(vertex)
#         center /= len(self.verts)
#
#         # Convert the angle from degrees to radians
#         angle_radians = np.radians(angle_degrees)
#
#         # Create the rotation matrix for Z-axis
#         rotation_matrix = np.array([
#             [np.cos(angle_radians), -np.sin(angle_radians), 0],
#             [np.sin(angle_radians), np.cos(angle_radians), 0],
#             [0, 0, 1]
#         ])
#
#         # Apply the rotation matrix to all vertices relative to the center
#         for i in range(self.nverts()):
#             # Translate the vertex to the origin
#             translated_vertex = self.verts[i] - center
#
#             # Convert the translated vertex to a NumPy array
#             vertex_array = np.array([translated_vertex[0], translated_vertex[1], translated_vertex[2]])
#
#             # Apply the rotation
#             rotated_vertex_array = np.dot(rotation_matrix, vertex_array)
#
#             # Update the vertex in the verts list
#             self.verts[i] = np.array([rotated_vertex_array[0], rotated_vertex_array[1], rotated_vertex_array[2]])
#
#         # Translate the model back to its original position
#         for i in range(self.nverts()):
#             self.verts[i] += center
#
#     def rotate(self, x, y, z):
#         self.rotate_x(x)
#         self.rotate_y(y)
#         self.rotate_z(z)


    ##############################





import sys
import numpy as np

class Model:

    def __init__(self, filename, material, x, y, z):
        self.verts = []
        self.faces = []
        self.material = material
        self.load_model(filename, x, y, z)

    def load_model(self, filename: str, x, y, z) -> None:
        try:
            with open(filename, 'r') as file:
                for line in file:
                    line = line.strip()
                    if line.startswith('v '):
                        coordinates = line[2:].split()
                        vertex = np.array([float(coordinates[0]) + x, float(coordinates[1]) + y, float(coordinates[2]) + z])
                        self.verts.append(vertex)
                    elif line.startswith('f '):
                        indices = line[2:].split()
                        face = np.array([int(indices[0]) - 1, int(indices[1]) - 1, int(indices[2]) - 1])
                        self.faces.append(face)
        except FileNotFoundError:
            print(f"Failed to open {filename}")
            sys.exit(1)

        print(f"Vertices: {len(self.verts)} Faces: {len(self.faces)}")

        min_vertex, max_vertex = self.get_bbox()
        #print(f"bbox: [{min_vertex} : {max_vertex}]")

    # def numpy_to_vec3f(self, np_array):
    #     return Vec3f(np_array[0], np_array[1], np_array[2])

    def ray_triangle_intersect(self, fi: int, orig: np.ndarray, dir: np.ndarray, tnear: float, tfar=float('inf')) -> bool:

        face_indices = self.faces[fi]

        # Get vertices for the current face
        v0 = self.verts[self.faces[fi][0]]
        v1 = self.verts[self.faces[fi][1]]
        v2 = self.verts[self.faces[fi][2]]

        edge1 = v1 - v0
        edge2 = v2 - v0

        # Begin calculating determinant - also used to calculate u parameter
        pvec = np.cross(dir, edge2)
        det = np.dot(edge1, pvec)

        if det < 1e-5:
            return False

        inv_det = 1.0 / det

        # Calculate distance from v0 to ray origin
        tvec = orig - v0

        # Calculate u parameter and test bound
        u = np.dot(tvec, pvec) * inv_det
        if u < 0.0 or u > 1.0:
            return False

        # Prepare to test v parameter
        qvec = np.cross(tvec, edge1)

        v = np.dot(dir, qvec)
        if v < 0 or u + v > det:
            return False

        # Calculate V parameter and test bound
        v = np.dot(dir, qvec) * inv_det
        if v < 0.0 or u + v > 1.0:
            return False

        # Calculate t, ray intersects triangle
        t = np.dot(edge2, qvec) * inv_det

        return (t > 1e-5) and (t < tfar)

    def nverts(self) -> int:
        return len(self.verts)

    def nfaces(self) -> int:
        return len(self.faces)

    def get_face(self, index: int) -> np.ndarray:
        assert 0 <= index < self.nfaces()
        return self.faces[index]

    def get_bbox(self) -> (np.ndarray, np.ndarray):
        min_vertex = max_vertex = self.verts[0]
        for vertex in self.verts[1:]:
            for i in range(3):
                min_vertex = np.array([min(min_vertex[0], vertex[0]), min(min_vertex[1], vertex[1]),
                                   min(min_vertex[2], vertex[2])])
                max_vertex = np.array([max(max_vertex[0], vertex[0]), max(max_vertex[1], vertex[1]),
                                   max(max_vertex[2], vertex[2])])
        return min_vertex, max_vertex

    def point(self, i: int) -> np.ndarray:
        assert 0 <= i < self.nverts()
        return self.verts[i]
    # def point(self, i: int) -> Vec3f:
    #     assert 0 <= i < self.nverts()
    #     return self.verts[i]

    def vert(self, fi: int, li: int) -> int:
        assert 0 <= fi < self.nfaces() and 0 <= li < 3
        face = self.faces[fi]
        if li == 0:
            return face[0]
        elif li == 1:
            return face[1]
        elif li == 2:
            return face[2]

    def __str__(self) -> str:
        output = ""
        for i in range(self.nverts()):
            output += f"v {self.point(i)}\n"
        for i in range(self.nfaces()):
            output += "f "
            for k in range(3):
                output += f"{self.vert(i, k) + 1} "
            output += "\n"
        return output

    def compute_normal(self, face):
        v0 = self.point(face[0])
        v1 = self.point(face[1])
        v2 = self.point(face[2])
        edge1 = v1 - v0
        edge2 = v2 - v0

        N = np.cross(edge1, edge2)
        Nmagnitude = np.linalg.norm(N)
        N = N / Nmagnitude if Nmagnitude != 0 else N

        return N

    def rotate_x(self, angle_degrees):
        center = np.array([0.0, 0.0, 0.0])
        for vertex in self.verts:
            center += np.array(vertex)
        center /= len(self.verts)

        # Convert the angle from degrees to radians
        angle_radians = np.radians(angle_degrees)

        # Create the rotation matrix for X-axis
        rotation_matrix = np.array([
            [1, 0, 0],
            [0, np.cos(angle_radians), -np.sin(angle_radians)],
            [0, np.sin(angle_radians), np.cos(angle_radians)]
        ])

        # Apply the rotation matrix to all vertices relative to the center
        for i in range(self.nverts()):
            # Translate the vertex to the origin
            translated_vertex = self.verts[i] - center

            # Convert the translated vertex to a NumPy array
            vertex_array = np.array([translated_vertex[0], translated_vertex[1], translated_vertex[2]])

            # Apply the rotation
            rotated_vertex_array = np.dot(rotation_matrix, vertex_array)

            # Update the vertex in the verts list
            self.verts[i] = np.array([rotated_vertex_array[0], rotated_vertex_array[1], rotated_vertex_array[2]])

        # Translate the model back to its original position
        for i in range(self.nverts()):
            self.verts[i] += center

    def rotate_y(self, angle_degrees):
        center = np.array([0.0, 0.0, 0.0])
        for vertex in self.verts:
            center += np.array(vertex)
        center /= len(self.verts)

        # Convert the angle from degrees to radians
        angle_radians = np.radians(angle_degrees)

        # Create the rotation matrix for Y-axis
        rotation_matrix = np.array([
            [np.cos(angle_radians), 0, np.sin(angle_radians)],
            [0, 1, 0],
            [-np.sin(angle_radians), 0, np.cos(angle_radians)]
        ])

        # Apply the rotation matrix to all vertices relative to the center
        for i in range(self.nverts()):
            # Translate the vertex to the origin
            translated_vertex = self.verts[i] - center

            # Convert the translated vertex to a NumPy array
            vertex_array = np.array([translated_vertex[0], translated_vertex[1], translated_vertex[2]])

            # Apply the rotation
            rotated_vertex_array = np.dot(rotation_matrix, vertex_array)

            # Update the vertex in the verts list
            self.verts[i] = np.array([rotated_vertex_array[0], rotated_vertex_array[1], rotated_vertex_array[2]])

        # Translate the model back to its original position
        for i in range(self.nverts()):
            self.verts[i] += center

    def rotate_z(self, angle_degrees):
        center = np.array([0.0, 0.0, 0.0])
        for vertex in self.verts:
            center += np.array(vertex)
        center /= len(self.verts)

        # Convert the angle from degrees to radians
        angle_radians = np.radians(angle_degrees)

        # Create the rotation matrix for Z-axis
        rotation_matrix = np.array([
            [np.cos(angle_radians), -np.sin(angle_radians), 0],
            [np.sin(angle_radians), np.cos(angle_radians), 0],
            [0, 0, 1]
        ])

        # Apply the rotation matrix to all vertices relative to the center
        for i in range(self.nverts()):
            # Translate the vertex to the origin
            translated_vertex = self.verts[i] - center

            # Convert the translated vertex to a NumPy array
            vertex_array = np.array([translated_vertex[0], translated_vertex[1], translated_vertex[2]])

            # Apply the rotation
            rotated_vertex_array = np.dot(rotation_matrix, vertex_array)

            # Update the vertex in the verts list
            self.verts[i] = np.array([rotated_vertex_array[0], rotated_vertex_array[1], rotated_vertex_array[2]])

        # Translate the model back to its original position
        for i in range(self.nverts()):
            self.verts[i] += center

    def rotate(self, x, y, z):
        self.rotate_x(x)
        self.rotate_y(y)
        self.rotate_z(z)