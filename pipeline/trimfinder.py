import numpy as np

import cv2
import random
import time
import math
from scipy.spatial import distance
from operator import attrgetter

from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .logging import log_image, im_logging_enabled
from .utils import convert_color
from .Line import Line, line_angle_difference
from .vanishingpointfinder import angle_with_vp

class TrimLine:
    def __init__(self, vp, rect, line_a, line_b):
        self.vp = vp
        self.rect = rect
        self.line_a = line_a
        self.line_b = line_b


    @property
    def inlier_lines(self):
        if self._inlier_lines is None:
            self._inlier_lines = list(map(lambda line_data: Line(line_data), self.inliers))

        return self._inlier_lines

class TrimFinder():
    def __init__(self, data, surface, vp):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface = surface
        self.vp = vp

    def solve(self, max_iterations=1000, max_time=0.25, angle_threshold=np.radians(7)):
        
        lines = sorted(self.surface.lines, key=attrgetter('length'), reverse=True)
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

            if line_angle_difference(line_a.angle, line_b.angle) > angle_threshold: continue

            directions = np.array([line_a.direction, line_b.direction]) 
            directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
            locations = np.array([line_a.midpoint, line_b.midpoint])

            angles = angle_with_vp(self.vp.model, locations, directions)

            if angles[0] < theta_thresh or angles[1] < theta_thresh:
                continue

            points = np.array([line_a.point_a, line_a.point_b, line_b.point_a, line_b.point_b])
            rect = cv2.minAreaRect(points)

            center = rect[0]
            size = rect[1]
            width = min(size[0], size[1])

            if width < min_width or width > max_width:
                continue

            trim_lines.append(TrimLine(self.vp, rect, line_a, line_b))


        # for line in self.surface.horizontal_vp[0].inlier_lines:
        #     surface_barriers.extend(line)
            
        return trim_lines
        
            
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
        self.surfaces = self.room.get_surfaces(surfaceTypes=[SurfaceType.Wall, SurfaceType.WallLike, SurfaceType.Floor, SurfaceType.Ceiling])

        for surface in self.surfaces:
            surface.horizontal_trim_lines = surface.vertical_trim_lines = []

            if surface.horizontal_vp is not None and len(surface.horizontal_vp) > 0:
                tf = TrimFinder(self.data, surface, surface.horizontal_vp[0])
                surface.horizontal_trim_lines = tf.solve()

            if surface.vertical_vp is not None and len(surface.vertical_vp) > 0:
                tf = TrimFinder(self.data, surface, surface.vertical_vp[0])
                surface.vertical_trim_lines = tf.solve()

        if im_logging_enabled(self.data):
            log_image(self.data, "trim.png", self.get_debug_image())


    def get_debug_image(self):

        hues = random.sample(range(0, 360), len(self.surfaces))
                    
        img = self.image.copy()

        Line.draw_all(img, self.data["lines"], color=(50,50,50), thickness=1)

        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            color = convert_color((hues[i],127,255), cv2.COLOR_HSV2RGB_FULL)

            if surface.horizontal_trim_lines is not None and len(surface.horizontal_trim_lines) > 0:

                for trim in surface.horizontal_trim_lines:
                    #trim = surface.horizontal_trim_lines[0]
                    
                    trim.line_a.draw(img, color=color, thickness=2)
                    trim.line_b.draw(img, color=color, thickness=2)


        return img


