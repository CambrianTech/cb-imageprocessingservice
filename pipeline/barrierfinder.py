import numpy as np
from scipy import ndimage
import cv2
import random
import time
from scipy.spatial import distance
import math

from .ade20k import ADE20K
from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .utils import resize_array, random_color, overlay_mask, normalize, convert_color
from .planegeometry import Dimension
from .extractsurfaces import box_like, legged_objects
from .vanishingpointfinder import draw_vp, angle_with_vp
from .logging import log_image, log_segmentation_image, im_logging_enabled
from cambrian.LineFunctions import LineFunctions
from .Line import line_angle_difference, Line, on_image_edge
from .room import Room, Surface

class Barrier():
    def __init__(self, line, search_width=6, length_multiplier=3.0):
        self.line = line
        self.source_lines = [line]
        self.indices = [line.id]
        self.search_width = search_width
        self.length_multiplier = length_multiplier

    def intersects(self, line):
        if line.group != self.line.group:
            return False

        rect_a = self.line.bounding_box(self.search_width, length_multiplier=self.length_multiplier)
        rect_b = line.bounding_box(self.search_width, length_multiplier=self.length_multiplier)
        result, _ = cv2.rotatedRectangleIntersection(rect_a, rect_b)

        return result != 0

    def merge(self, line):
        line_data = LineFunctions.merge_lines((self.line.point_a, self.line.point_b), (line.point_a, line.point_b))
        self.line = Line(np.array([(line_data[0][0], line_data[0][1], line_data[1][0], line_data[1][1])], dtype=np.int).reshape(4))
        self.source_lines.append(line)

    def extend_to(self, line):
        print("Extend to point")

class BarrierFinder():
    def __init__(self, data, surface, hed):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface = surface
        self.hed = hed

    def solve(self, angle_threshold=np.radians(5), alter_angle_threshold=np.radians(7), min_length=50, max_length=1000):
        
        if len(self.surface.horizontal_vp) == 0 or len(self.surface.vertical_vp) == 0:
            return []

        horizontal_vp = self.surface.horizontal_vp[0]
        vertical_vp = self.surface.vertical_vp[0]

        candidates = []
        vp_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        theta_thresh = np.cos(angle_threshold)

        test_length = 5

        min_length_sq = min_length * min_length
        max_length_sq = max_length * max_length

        barriers = []
        group = -1
        for poly in self.surface.polygons:
            num_pts = len(poly)
            last_line = None
            group += 1

            for i in range(num_pts):
                point_a = poly[i][0]
                point_b = poly[(i+1) % num_pts][0]

                length_sq = distance.sqeuclidean(point_a, point_b)

                if length_sq < min_length_sq or length_sq > max_length_sq:
                    continue

                if on_image_edge(point_a, self.image) and on_image_edge(point_b, self.image):
                    continue
                
                line = Line(np.array([point_a[0], point_a[1], point_b[0], point_b[1]]))

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
                    line = Line(np.array([line.midpoint[0] - direction[0], line.midpoint[1] - direction[1], line.midpoint[0] + direction[0], line.midpoint[1] + direction[1]], dtype=np.int).reshape(4), group=group)

                    match = next(filter(lambda x: x.intersects(line), barriers), None)

                    if match is not None:
                        if LineFunctions.line_angle_difference(line.angle, match.line.angle) < alter_angle_threshold:
                            match.merge(line)
                            continue
                        else:
                            match.extend_to(line)
                            group += 1
                    
                    barriers.append(Barrier(line))

                    last_line = line

                elif last_line is not None:
                    angle = line_angle_difference(line.angle, last_line.angle)

                

                
        return barriers
        

class LegFinder():
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        

    def solve(self, max_iterations=3000):

        self.surfaces = self.room.get_surfaces(surfaceTypes=[SurfaceType.Other, SurfaceType.Floor])
        
        for surface in self.surfaces:

            surface.barriers = surface.border_lines
            
            
        # start_time = time.time()
        # for ransac_iter in range(max_iterations):
        #     if time.time() - start_time > max_time:
        #         break

            


        
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

        hed = cv2.resize(self.data["hed"], (self.image.shape[1], self.image.shape[0]))
        hed = cv2.normalize(hed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        self.surfaces = []
        self.surfaces.extend(self.room.get_surfaces(surfaceTypes=[SurfaceType.Wall]))
        self.surfaces.extend(self.room.get_surfaces(labels=box_like))

        for surface in self.surfaces:
            bf = BarrierFinder(self.data, surface, hed)
            min_size = np.sqrt(surface.min_area) / 2
            max_size = np.sqrt(surface.max_area) * 2
            surface.barriers = bf.solve(min_length=min_size, max_length=max_size)

        # lf = LegFinder(self.data)
        # lf.solve()

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

        for i in range(len(self.room.surfaces)):
            surface = self.room.surfaces[i]
            color = convert_color((hues[i],127,255), cv2.COLOR_HSV2RGB_FULL)

            for poly in surface.contours:
                num_pts = len(poly)
                for i in range(num_pts):
                    point_a = poly[i][0]
                    point_b = poly[(i+1) % num_pts][0]

                    line = Line(np.array([point_a[0], point_a[1], point_b[0], point_b[1]]))

                    line.draw(img, color=color)

            for barrier in surface.barriers:
                barrier.line.draw(img, color=color, thickness=2)

            for barrier in surface.barriers:
                cv2.drawMarker(img, barrier.line.point_a, color=color)
                cv2.drawMarker(img, barrier.line.point_b, color=color)

        return img


