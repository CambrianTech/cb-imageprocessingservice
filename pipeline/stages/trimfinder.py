import numpy as np

import cv2
import random
import time
import math
from scipy.spatial import distance
from operator import attrgetter

from cambrian.LineFunctions import LineFunctions
from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_image, im_logging_enabled
from pipeline.misc.utils import convert_color
from pipeline.components.line import draw_lines
from .vanishingpointfinder import angle_with_vp
from pipeline.misc.utils import random_color

class TrimLine:
    def __init__(self, vp, rect, line_a, line_b):
        self.vp = vp
        self.rect = rect
        self.line_a = line_a
        self.line_b = line_b

class TrimFinder():
    def __init__(self, data, surface, barrier):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface = surface
        self.barrier = barrier

    def solve(self, max_iterations=1000, max_time=0.25, angle_threshold=np.radians(7)):
        
        lines = sorted(self.barrier.lines, key=attrgetter('length'), reverse=True)
        num_lines = len(lines)

        if num_lines < 2:
            return []

        ideal_width = self.image.shape[0] * 0.04

        min_width = ideal_width / 2
        max_width = ideal_width * 2

        max_midpoint_distance_sq = ideal_width * 5
        max_midpoint_distance_sq *= max_midpoint_distance_sq

        start_time = time.time()

        trim_lines = []

        theta_thresh = np.cos(angle_threshold)

        for ransac_iter in range(max_iterations):
            if time.time() - start_time > max_time:
                break

            i = random.randint(0, num_lines-1)
            j = random.randint(0, num_lines-1)
            
            if i == j: continue

            line_a = lines[i]
            line_b = lines[j]

            if LineFunctions.line_angle_difference(line_a.angle, line_b.angle) > angle_threshold: continue

            directions = np.array([line_a.direction, line_b.direction]) 
            directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
            locations = np.array([line_a.midpoint, line_b.midpoint])

            angles = angle_with_vp(self.barrier.vp.model, locations, directions)

            if angles[0] < theta_thresh or angles[1] < theta_thresh:
                continue

            points = np.array([line_a.point_a, line_a.point_b, line_b.point_a, line_b.point_b])
            rect = cv2.minAreaRect(points)

            center = rect[0]
            size = rect[1]
            width = min(size[0], size[1])

            if width < min_width or width > max_width:
                continue

            trim_lines.append(TrimLine(self.barrier.vp, rect, line_a, line_b))
            
        
        
            
class PipelineTrimFinder(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.FindTrim

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

        #detect trim around edges and (1/3rd of center horizontal, around 1 meter high) of walls using horizontal vp inliers.
        #create segmentation category?
        self.surfaces = list(filter(lambda s: s.barriers is not None, self.room.surfaces))

        self.trim_lines = []

        for surface in self.surfaces:
            for barrier in surface.barriers:
                tf = TrimFinder(self.data, surface, barrier)
                tf.solve()

        if im_logging_enabled(self.data):
            log_image(self.data, "trim.png", self.get_debug_image())


    def get_debug_image(self):

        hues = random.sample(range(0, 360), len(self.surfaces))
                    
        img = self.image.copy()

        draw_lines(img, self.data["lines"], color=(50,50,50), thickness=1)

        # for vp in self.room.horizontal_vps:
        #     draw_lines(img, vp.inliers, color=random_color(), thickness=2)

        for surface in self.surfaces:
            for barrier in surface.barriers:
                draw_lines(img, barrier.lines, color=random_color(), thickness=2)

        # for trim in self.trim_lines:
        #     draw_lines(img, [trim.line_a, trim.line_b], color=random_color(), thickness=2)

        return img


