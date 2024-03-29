import argparse
import sys
from PIL import Image
import math
from model import Model
import time
# import numpy as np
from sphere import *

envmap_width = 0
envmap_height = 0
envmap = []

front_img = []
front_img_width = 0
front_img_height = 0

global image_distance
image_distance = -5

#comment
class light:
    def __init__(self, position, intensity):
        self.position = position
        self.intensity = intensity

class Material:
    def __init__(self, n=None, a=None, color=None, spec=None):
        if n is None:
            self.refractive_index = 1.0
        else:
            self.refractive_index = n

        if a is None:
            self.albedo = np.array([1, 0, 0, 0])
        else:
            self.albedo = a

        if color is None:
            self.diffuse_color = (0, 0, 0)
        else:
            self.diffuse_color = color

        if spec is None:
            self.specular_exponent = 0.0
        else:
            self.specular_exponent = spec


def reflect(I, N):
    return I - N * 2.0 * np.dot(I, N)

def refract(I, N, refractive_index):
    cosi = -max(-1.0, min(1.0, np.dot(I, N)))
    etai = 1.0
    etat = refractive_index
    n = N

    if cosi < 0:
        cosi = -cosi
        etai, etat = etat, etai
        n = -N

    eta = etai / etat
    k = 1 - eta * eta * (1 - cosi * cosi)

    if k < 0:
        return np.array([0, 0, 0])
    else:
        return I * eta + n * (eta * cosi - math.sqrt(k))

def scene_intersect(orig, dir, spheres, renderModel):
    spheres_dist = math.inf
    hit = None
    N = None
    material = Material()  # Assign a default Material object

    for i in range(len(spheres)):
        dist_i = [0.0]
        if spheres[i].ray_intersect(orig, dir, dist_i) and dist_i[0] < spheres_dist:
            spheres_dist = dist_i[0]
            hit = orig + dir * dist_i[0]

            N = hit - spheres[i].center
            Nlength = np.linalg.norm(N)
            N = N / Nlength if Nlength != 0 else N

            material = spheres[i].material

    if renderModel is not None:
        obj_dist = float('inf')  # Initialize obj_dist with infinity
        for fi in range(renderModel.nfaces()):
            tnear = [0.0]  # Create a list to store the intersection distance
            if renderModel.ray_triangle_intersect(fi, orig, dir, tnear[0]):
                face = renderModel.get_face(fi)
                if tnear[0] < obj_dist:  # Check if the new intersection is closer
                    obj_dist = tnear[0]  # Update the obj_dist variable
                    hit = orig + dir * tnear[0]
                    N = renderModel.compute_normal(face)  # Compute the surface normal for the triangle
                    material = renderModel.material

    return hit, N, material

def cast_ray(orig, dir, spheres, lights, renderModel, zoom_factor, maxDepth, depth=0):

    point, N, material = scene_intersect(orig, dir, spheres, renderModel)

    if depth > maxDepth or point is None:
        dir_normalized = dir / np.linalg.norm(dir)

        # Compute the aspect ratio of the input image
        aspect_ratio = front_img_width / front_img_height

        u = 0.5 * (dir_normalized[0] / (dir_normalized[2] * aspect_ratio) + 1.0)
        v = 0.5 * (dir_normalized[1] / dir_normalized[2] + 1.0)

        # Apply a scaling factor to zoom out the background image
        u = u * zoom_factor
        v = v * zoom_factor

        # Center the image by adding the necessary offset after the zoom
        u_offset = (1 - zoom_factor) / 2
        v_offset = (1 - zoom_factor) / 2
        u = u + u_offset
        v = v + v_offset

        # Check if the scaled coordinates are out of the image bounds
        if u < 0 or u > 1 or v < 0 or v > 1:
            return np.array([0, 0, 0])

        # Scale u and v to image dimensions
        x = int(u * front_img_width)
        y = int(v * front_img_height)

        # Clamp to bounds of the image
        x = max(0, min(front_img_width - 1, x))
        y = max(0, min(front_img_height - 1, y))

        # Fetch the color from the front image
        background_color = front_img[x + y * front_img_width]
        return background_color


    reflect_dir = dir - 2.0 * np.dot(dir, N) * N
    reflectMagnitude = np.linalg.norm(reflect_dir)
    reflect_dir = reflect_dir / reflectMagnitude if reflectMagnitude != 0 else reflect_dir

    refract_dir = refract(dir, N, material.refractive_index)
    refractMagnitude = np.linalg.norm(refract_dir)
    refract_dir = refract_dir / refractMagnitude if refractMagnitude != 0 else refract_dir

    reflect_orig = point - N * 1e-3 if np.dot(reflect_dir, N) < 0 else point + N * 1e-3
    refract_orig = point - N * 1e-3 if np.dot(refract_dir, N) < 0 else point + N * 1e-3

    reflect_color = cast_ray(reflect_orig, reflect_dir, spheres, lights, renderModel, zoom_factor, maxDepth, depth + 1)
    refract_color = cast_ray(refract_orig, refract_dir, spheres, lights, renderModel, zoom_factor, maxDepth, depth + 1)

    diffuse_light_intensity = 0
    specular_light_intensity = 0

    for light in lights:

        light_distance = np.linalg.norm(light.position - point) # Magnitude
        light_dir = (light.position - point) / light_distance if light_distance != 0 else (light.position - point)
        shadow_orig = point - N * 1e-3 if np.dot(light_dir, N) < 0 else point + N * 1e-3 # checking if the point lies in the shadow of lights[i]
        shadow_pt, shadow_N, tmpmaterial = scene_intersect(shadow_orig, light_dir, spheres, renderModel)
        if shadow_pt is not None and np.linalg.norm(shadow_pt - shadow_orig) < light_distance:
            continue
        diffuse_light_intensity += light.intensity * max(0.0, np.dot(light_dir, N))
        reflected_dir = reflect(-light_dir, N)
        specular_intensity = np.power(np.maximum(0.0, -np.dot(reflected_dir, dir)), material.specular_exponent)
        specular_light_intensity += specular_intensity * light.intensity

    diffuse_color = np.array(material.diffuse_color)  # Convert tuple to np array
    albedo_0, albedo_1, albedo_2, albedo_3 = material.albedo  # Unpack albedo components

    return diffuse_color * diffuse_light_intensity * albedo_0 + np.array([1.0, 1.0,
                                                                      1.0]) * specular_light_intensity * albedo_1 + reflect_color * albedo_2 + refract_color * albedo_3

def render(imgSize, spheres, lights, renderModel, zoomFactor, maxDepth):

    print('Rendering Started')
    renderStartTime = time.time()

    width, height = imgSize

    total_pixels = width * height
    pixels_processed = 0

    fov = math.pi / 2.0 # Field of View of the camera
    framebuffer = [np.array([0, 0, 0])] * (width * height)

    for j in range(height):
        for i in range(width):
            x = (2 * (i + 0.5) / float(width) - 1) * math.tan(fov / 2.0) * width / float(height)
            y = -(2 * (j + 0.5) / float(height) - 1) * math.tan(fov / 2.0)

            dir = np.array([x, y, -1], dtype=np.float64)
            magnitude = np.linalg.norm(dir)
            dir = dir / magnitude if magnitude != 0 else dir

            framebuffer[i + j * width] = cast_ray(np.array([0, 0, 0]), dir, spheres, lights, renderModel, zoomFactor, maxDepth)

            pixels_processed += 1
            progress = pixels_processed / total_pixels
            hashes = '#' * int(progress * 10)
            spaces = '-' * (10 - len(hashes))
            percentage = progress * 100
            sys.stdout.write(f"\rRendering Progress: {percentage:.2f}% [{hashes}{spaces}] {pixels_processed}/{total_pixels} rays processed")
            sys.stdout.flush()

    image = Image.new("RGB", (width, height))

    for j in range(height):
        for i in range(width):
            pixel_color = framebuffer[i + j * width]
            r, g, b = int(255 * max(0, min(1, pixel_color[0]))), int(255 * max(0, min(1, pixel_color[1]))), int(
                255 * max(0, min(1, pixel_color[2]))
            )
            image.putpixel((i, j), (r, g, b))

    print("")
    print('Render Ended')

    renderEndTime = time.time()
    renderTime = (renderEndTime - renderStartTime) / 60
    print(f"Render Time: {renderTime: .3f} mins")
    image.save("out.png", "PNG")

def load_front_image(filename):
    global front_img, front_img_width, front_img_height
    try:
        image = Image.open(filename)
        image = image.convert("RGB")  # Ensure image is in RGB format
        front_img_width, front_img_height = image.size
        front_img = np.array(image).reshape(-1, 3) / 255.0  # Flatten the image array and normalize
    except IOError:
        sys.stderr.write("Error: cannot load the front image\n")
        sys.exit(-1)


def main(args):

    mainStartTime = time.time()
    print('Main program started')

    frontImg = args.frontImg
    modelName = args.modelName
    maxDepth = args.maxDepth
    modelx, modely, modelz = args.modelx, args.modely, args.modelz
    modelxR, modelyR, modelzR = args.modelxR, args.modelyR, args.modelzR
    imgSize = (args.imgWidth, args.imgHeight)
    zoomFactor = args.zoomFactor

    load_front_image(frontImg)

    ivory = Material(1.0, (0.6, 0.3, 0.1, 0.0), (0.4, 0.4, 0.3), 50.0)
    glass = Material(1.5, (0.0,  0.5, 0.1, 0.8), (0.6, 0.7, 0.8), 125.0)
    red_rubber = Material(1.0, (0.9, 0.1, 0.0, 0.0), (0.3, 0.1, 0.1), 10.0)
    mirror = Material(1.0, (0.0, 10.0, 0.8, 0.0), (1.0, 1.0, 1.0), 1425.0)
    grayEasy = Material(1.0, (1.0, 0.0, 0.0, 0.0), (0.5, 0.5, 0.5), 0.0)

    spheres = []
    # spheres.append(Sphere(np.array([0.0, 0.0, -18]), 4, glass))

    lights = []
    lights.append(light(np.array([-20, 20, 20]), 1.5))
    lights.append(light(np.array([30, 50, -25]), 1.8))
    lights.append(light(np.array([30, 20, 30]), 1.7))

    print('Lights, and Materials Loaded')

    load_model = False  # Set this to False to skip loading the 3D model
    renderModel = None

    if load_model:
        print("Loading 3d Model")
        modelStartTime = time.time()
        # Arguments are name of obj File, material, x, y , z (coordinates)
        renderModel = Model(modelName, glass, modelx, modely, modelz)

        # Rotation around x, y, and z axis (degrees) CCW
        # X axis -> Left to right horizontal on the screen
        # Y axis -> Bottom to top vertical on the screen
        # Z axis -> Out of the screen
        renderModel.rotate(modelxR, modelyR, modelzR)
        print("3D Model Created")
        modelEndTime = time.time()
        modelExecutionTime = (modelEndTime - modelStartTime) / 60.0
        print(f"3D Model Creation Time: {modelExecutionTime: .3f} mins")


    render(imgSize, spheres,lights, renderModel, zoomFactor, maxDepth)

    mainEndTime = time.time()  # Stop measuring the execution time
    mainExecutionTime = (mainEndTime - mainStartTime) / 60.0
    print(f"Total Execution Time: {mainExecutionTime: .3f} mins")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render a 3D scene with ray tracing.")
    parser.add_argument("--frontImg", type=str, default="tree.jpg", help="Path to the front image.")
    parser.add_argument("--modelName", type=str, default="objFiles/prism.obj", help="Path to the 3D model file.")
    parser.add_argument("--maxDepth", type=int, default=4, help="Maximum depth for ray tracing.")
    parser.add_argument("--modelx", type=float, default=0, help="Model x coordinate.")
    parser.add_argument("--modely", type=float, default=0, help="Model y coordinate.")
    parser.add_argument("--modelz", type=float, default=-18, help="Model z coordinate.")
    parser.add_argument("--modelxR", type=float, default=90, help="Model rotation around the x-axis.")
    parser.add_argument("--modelyR", type=float, default=45, help="Model rotation around the y-axis.")
    parser.add_argument("--modelzR", type=float, default=0, help="Model rotation around the z-axis.")
    parser.add_argument("--imgWidth", type=int, default=1024, help="Width of the output image.")
    parser.add_argument("--imgHeight", type=int, default=768, help="Height of the output image.")
    parser.add_argument("--zoomFactor", type=float, default=2, help="Zoom factor for the camera.")

    args = parser.parse_args()

    main(args)