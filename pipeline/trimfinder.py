import numpy as np

import cv2
import random
import time
import math
from operator import attrgetter

from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .logging import log_image, im_logging_enabled
from .utils import convert_color
from .Line import Line

class TrimLine:
    def __init__(self, vp, primary_line, lines):
        self.vp = vp
        self.primary_line = primary_line
        self.lines = lines

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

    def solve(self, max_iterations=500, max_time=0.25):
        
        seeds = sorted(self.vp.inlier_lines, key=attrgetter('length'), reverse=True)
        lines = sorted(self.surface.lines, key=attrgetter('length'), reverse=True)

        if len(seeds) == 0:
            seeds = lines

        num_seeds = len(seeds)
        num_lines = len(lines)

        if num_seeds < 1 or num_lines < 1:
            return []

        start_time = time.time()

        for ransac_iter in range(max_iterations):
            if time.time() - start_time > max_time:
                break

            ind1 = random.randint(0, num_seeds-1)
            ind2 = random.randint(0, num_lines-1)


        # for line in self.surface.horizontal_vp[0].inlier_lines:
        #     surface_barriers.extend(line)
            
        return seeds
        
            
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
            surface.trim_lines = []

            if surface.horizontal_vp is not None and len(surface.horizontal_vp) > 0:
                tf = TrimFinder(self.data, surface, surface.horizontal_vp[0])
                surface.trim_lines.extend(tf.solve())

            if surface.vertical_vp is not None and len(surface.vertical_vp) > 0:
                tf = TrimFinder(self.data, surface, surface.vertical_vp[0])
                surface.trim_lines.extend(tf.solve())

        if im_logging_enabled(self.data):
            log_image(self.data, "trim.png", self.get_debug_image())


    def get_debug_image(self):

        hues = random.sample(range(0, 360), len(self.surfaces))
                    
        img = self.image.copy()

        Line.draw_all(img, self.data["lines"], color=(0,255,0), thickness=1)

        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            color = convert_color((hues[i],127,255), cv2.COLOR_HSV2RGB_FULL)

            if surface.trim_lines is not None:
                Line.draw_all(img, surface.trim_lines, color=color, thickness=2)


        return img


