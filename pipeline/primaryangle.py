from pipeline.core import PipelineStep
import cv2
import numpy as np
import math
from cambrian import image_processing as ip, transformations as T, geometry as geo

# Z is UP
rotX = T.rotation_matrix(0.00, [1, 0, 0])
rotY = T.rotation_matrix(0.00, [0, 1, 0])
rotZ = T.rotation_matrix(-0.06, [0, 0, 1])

r_range = g_range = b_range = 255

RIGHT_ANGLE = (math.pi/2.0)  # 90 degrees


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

    print("Floor is %.2f degrees from UP" % geo.radians_to_degrees(best_angle))
    return surface_index, most_pixels, best_intersection


def get_candidate_walls(floor_normal, isolated_surfaces, maxAngle=20):
    candidate_walls = []
    max_diff = geo.degrees_to_radians(maxAngle)

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

            print("%d) candidate wall normal = %s, angle diff = %.2f, color = %s, score=%f"
                  % (wall_count, normal, angle_diff, surface[0], score))

    wall_count = wall_count + 1

    if len(candidate_walls) == 0:
        best_wall_index = 0
        candidate_walls.append(isolated_surfaces[best_wall_index])

    return candidate_walls, best_wall_index


class PipelineDeterminePrimaryAngles(PipelineStep):

    @property
    def required_keys(self) -> list:
        return ["semantic_probs", "normals", "elevation"]

    @property
    def output_keys(self) -> list:
        return ["kmeans_normals", "camera_rotation", "camera_elevation", "floor_rotation"]

    def run(self, data):

        mask = np.uint8(255*data["semantic_probs"][:, :, 0])
        normals = data["normals"]
        elevation = data["elevation"]

        kmeans, labels, centers = ip.kmeans_image(255*normals, 5)

        data["kmeans_normals"] = kmeans

        reduced_normals = kmeans
        reduced_mask = mask
        reduced_mask[reduced_mask > 127] = 255
        reduced_mask[reduced_mask < 255] = 0

        # build surfaces
        isolated_surfaces = []
        for color in centers:
            normal = get_normal_from_rgb(color)
            color_mask = ip.isolate_color(reduced_normals, color)
            isolated_surfaces.append((color, normal, color_mask))

        floor_materials = [255]  # floor, rug

        # find floor
        floor_index, _, floor_intersection = get_matching_surface(
            reduced_mask, isolated_surfaces, floor_materials)

        if floor_index < 0:
            print("Invalid surfaces")
            return None

        floor_surface = isolated_surfaces[floor_index]

        # Calculate the camera pitch and roll from the floor normal.
        # Get the floor normal by taking the normals at the 100 pixels
        # most likely to be floor and average them.

        result_prob = cv2.bitwise_and(mask, mask, mask=floor_intersection)

        floor_indices = np.stack(np.unravel_index(
            np.argsort(-result_prob.flatten()), result_prob.shape), axis=-1)
        floor_indices = floor_indices[:100]

        strongest_floor = normals[floor_indices[:, 0], floor_indices[:, 1]]

        floor_normal = np.mean(strongest_floor, axis=0) - 0.5
        floor_normal_len = max(0.00001, np.linalg.norm(floor_normal))
        floor_normal /= floor_normal_len
        cam_pitch = math.asin(floor_normal[1])
        cam_roll = math.asin(floor_normal[0])
        data["camera_rotation"] = [cam_pitch, 0.0, cam_roll]
        
        print("camera rotation: " + str(data["camera_rotation"]))
        
        floor_distances = elevation[floor_indices[:,
                                                  0], floor_indices[:, 1]].flatten()
        floor_distances.sort()

        floor_elevation_pixels = 127.5 - \
            float(floor_distances[np.floor_divide(len(floor_distances), 2)])
        pixels_per_meter = 127.5 / 300.0
        floor_elevation = floor_elevation_pixels / pixels_per_meter

        floor_elevation = np.clip(floor_elevation, 80.0, 170.0)  # valid range

        data["camera_elevation"] = floor_elevation / 100.0
        
        print("floor elevation: " + str(data["camera_elevation"]))

        # find candidate wall surfaces
        candidate_walls, primary_wall_index = get_candidate_walls(
            floor_surface[1], isolated_surfaces)

        # calculate floor rotation:
        floor_rotation = 0.0

        x_unit_normal = [1, 0, 0]
        y_unit_normal = [0, 1, 0]
        z_unit_normal = [0, 0, 1]

        floor_angle_x = math.pi - \
            geo.angle_between(floor_surface[1], y_unit_normal)

        if primary_wall_index >= 0:
            floor_rotation = geo.angle_between(
                candidate_walls[primary_wall_index][1], x_unit_normal)

        data["floor_rotation"] = floor_rotation
