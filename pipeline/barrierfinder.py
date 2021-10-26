import numpy as np
from scipy import ndimage
import cv2
from skimage.morphology import remove_small_objects
import random
import time
from scipy.spatial import distance
import math

from .ade20k import ADE20K
from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .utils import resize_array, random_color, overlay_mask, normalize
from .planegeometry import Dimension
from .extractsurfaces import box_like
from .vanishingpointfinder import draw_vp, angle_with_vp
from .logging import log_image, log_segmentation_image, im_logging_enabled
from cambrian.LineFunctions import LineFunctions
from .Line import line_angle_difference, Line, on_image_edge
from .room import Room, Surface

class BarrierFinder():
    def __init__(self, data, surface, hed):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface = surface
        self.hed = hed

    def solve(self, max_iterations=3000, max_time=0.5, min_distance=50, max_distance=1000, angle_threshold=np.radians(3), min_vp_mean=0.1, min_hed_mean=0.15):
        
        if len(self.surface.horizontal_vp) == 0 or len(self.surface.vertical_vp) == 0:
            return []

        horizontal_vp = self.surface.horizontal_vp[0]
        vertical_vp = self.surface.vertical_vp[0]

        candidates = []
        vp_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        draw_vp(vp_mask, horizontal_vp, color=1, thickness=2)
        draw_vp(vp_mask, vertical_vp, color=1, thickness=2) 

        theta_thresh = np.cos(angle_threshold)

        points = []
        for poly in self.surface.polygons:
            for point in poly:
                points.append(point[0])

        for line in self.surface.border_lines:
            points.append(line.point_a)
            points.append(line.point_b)

        for line_data in horizontal_vp.inliers:
            points.append((line_data[0], line_data[1]))
            points.append((line_data[2], line_data[3]))

        if len(points) < 5:
            return []

        #max_iterations = min(len(points) * 20, max_iterations)

        start_time = time.time()

        min_distance_sq = min_distance * min_distance
        max_distance_sq = max_distance * max_distance

        for ransac_iter in range(max_iterations):
            if time.time() - start_time > max_time:
                break

            items = random.sample(points, 2)

            point_a = items[0]
            point_b = items[1]

            length_sq = distance.sqeuclidean(point_a, point_b)

            if length_sq < min_distance_sq or length_sq > max_distance_sq:
                continue

            line = Line(np.array([(point_a[0], point_a[1], point_b[0], point_b[1])], dtype=np.int).reshape(4))

            directions = np.array([line.direction]) 
            directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
            locations = np.array([line.midpoint])

            vert_theta = angle_with_vp(vertical_vp.model, locations, directions)
            horiz_theta = angle_with_vp(horizontal_vp.model, locations, directions)
            
            is_vertical = vert_theta > theta_thresh
            is_horizontal = horiz_theta > theta_thresh

            if is_vertical or is_horizontal:
                vp = horizontal_vp if is_horizontal else vertical_vp
                est_directions = locations - vp.direction

                direction = normalize(est_directions[0]) * 0.5 * line.length

                line = Line(np.array([line.midpoint[0] - direction[0], line.midpoint[1] - direction[1], line.midpoint[0] + direction[0], line.midpoint[1] + direction[1]], dtype=np.int).reshape(4))

                num_samples = int(min(max(line.length / 10, 5), 15))

                def under_threshold(img, threshold, func=np.mean):
                    if threshold < 0:
                        return False
                    samples = LineFunctions.get_line_samples(line.point_a, line.point_b, img, num_samples)
                    return func(samples) < threshold

                if under_threshold(self.hed, min_hed_mean) or under_threshold(vp_mask, min_vp_mean):
                    continue

                candidates.append(line)

        diagonal = math.hypot(self.image.shape[0], self.image.shape[1])
        candidates = Line.merge(candidates, search_width=diagonal/200, search_length=1.2)
        
        candidates.sort(key=lambda x:x.length, reverse=True)

        return candidates

        
class PipelineBarrierFinder(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Barriers

    @property
    def required_keys(self) -> list:
        return ["room", "downscaled", "isolated", "lines"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):

        self.data = data
        self.image = self.data["downscaled"]
        self.room = self.data["room"]

        self.found_lines = []
        hed = cv2.resize(self.data["hed"], (self.image.shape[1], self.image.shape[0]))
        hed = cv2.normalize(hed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        self.surfaces = []
        self.surfaces.extend(self.room.get_surfaces(surfaceTypes=[SurfaceType.Wall]))
        self.surfaces.extend(self.room.get_surfaces(labels=box_like))

        for surface in self.surfaces:
            rf = BarrierFinder(self.data, surface, hed)
            min_size = np.sqrt(surface.min_area) / 2
            max_size = np.sqrt(surface.max_area) * 2
            lines = rf.solve(min_distance=min_size, max_distance=max_size)
            self.found_lines.extend(lines)

        if im_logging_enabled(self.data):
            log_image(self.data, "barriers.png", self.get_debug_image())


    def get_debug_image(self):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.room.surfaces))

        #overlay probs
        for i in range(len(self.room.surfaces)):
            surface = self.room.surfaces[i]
            mask = surface.mask > 0

            max_value = 0.9
            if max_value > 0:
                img_hsv[:, :, 0][mask] = hues[i]
                img_hsv[:, :, 1][mask] = 255 * np.power(surface.probs[mask], 0.15)
                    
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)

        for line in self.found_lines:

            line.draw(img, color=(255,255,0), thickness=1)            

        return img


