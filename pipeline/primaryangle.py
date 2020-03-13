from pipeline.core import PipelineStep
import cv2
import numpy as np
import math
from cambrian import image_processing as ip, transformations as T, geometry as geo

# Z is UP
rotX = T.rotation_matrix(0.00, [1, 0, 0])
rotY = T.rotation_matrix(0.00, [0, 1, 0])
rotZ = T.rotation_matrix(0.00, [0, 0, 1])

r_range = g_range = b_range = 255

RIGHT_ANGLE = math.pi / 2.0  # 90 degrees


def scale_component(color, range=255.0):
    return 2.0 * (float(color) / float(range)) - 1.0


def get_normal_from_rgb(rgb):
    x = scale_component(rgb[0], r_range)
    y = scale_component(rgb[1], g_range)
    z = scale_component(rgb[2], b_range)

    return rotate_normal((x, y, z))


def rotate_normal(normal):
    result = np.dot(normal, rotX[:3, :3].T)
    result = np.dot(result, rotY[:3, :3].T)
    result = np.dot(result, rotZ[:3, :3].T)
    return tuple(result)


def get_matching_surface(reduced_mask, isolated_surfaces, isolated_values, ignore_indices=[], max_angle=40.0):
    isolated_mask = np.zeros(reduced_mask.shape, dtype=np.uint8)

    for value in isolated_values:
        isolated_mask[reduced_mask == value] = 255

    most_pixels = 0
    surface_index = -1
    best_intersection = 0
    best_angle = 0

    straight_up = [0, 0, 1]

    max_radians = geo.degrees_to_radians(max_angle)

    # find best matching surface
    for i, surface in enumerate(isolated_surfaces):
        if i not in ignore_indices:
            normal = surface[1]
            angle = geo.angle_between(normal, straight_up)

            intersection = cv2.bitwise_and(surface[2], isolated_mask)

            intersection_pixels = cv2.countNonZero(intersection)

            if intersection_pixels > most_pixels and angle < max_radians:
                most_pixels = intersection_pixels
                surface_index = i
                best_intersection = intersection
                best_angle = angle

    if surface_index == -1 and max_angle != 180.0:
        return get_matching_surface(reduced_mask, isolated_surfaces, isolated_values, ignore_indices, max_angle=180.0)

    return surface_index, most_pixels, best_intersection


def get_candidate_walls(floor_normal, isolated_surfaces, max_angle=20):
    candidate_walls = []
    max_diff = geo.degrees_to_radians(max_angle)

    kernel = np.ones((3, 3), np.uint8)
    best_score = 0
    best_wall_index = 0
    wall_count = 0

    overall_best_diff = math.pi

    for i, surface in enumerate(isolated_surfaces):
        normal = surface[1]
        angle = geo.angle_between(normal, floor_normal)
        vert_diff = abs(angle - RIGHT_ANGLE)

        if vert_diff < overall_best_diff:
            overall_best_diff = vert_diff

        if vert_diff < max_diff:
            candidate_walls.append(surface)
            angle_diff = 180.0 * vert_diff / math.pi

            opening = cv2.morphologyEx(surface[2], cv2.MORPH_OPEN, kernel)
            total_pixels = cv2.countNonZero(opening)

            score = total_pixels / math.sqrt(angle_diff + 3.0)

            if score > best_score:
                best_wall_index = wall_count
                best_score = total_pixels

            wall_count = wall_count + 1

    if len(candidate_walls) == 0:
        best_wall_index = 0
        candidate_walls.append(isolated_surfaces[best_wall_index])

    return candidate_walls, best_wall_index


class PipelineDeterminePrimaryAngles(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["semantic_probs"]

    @property
    def output_keys(self) -> list:
        return ["camera_rotation", "camera_elevation", "floor_rotation"]

    def run(self, data):

        cam_pitch = -0.2
        cam_yaw = 0.0
        cam_roll = 0.0

        data["camera_rotation"] = [cam_pitch, cam_yaw, cam_roll]

        print("camera rotation:", data["camera_rotation"])

        data["camera_elevation"] = 1.3

        print("floor elevation:", data["camera_elevation"])

        data["floor_rotation"] = 0.0

        print("floor rotation:", data["floor_rotation"])
