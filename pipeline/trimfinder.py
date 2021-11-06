import numpy as np

import cv2
import random
import time
import math

from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .logging import log_image, im_logging_enabled
from .utils import convert_color
from .Line import Line

class TrimFinder():
    def __init__(self, data, surface):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface = surface

    def solve(self):
        
        if self.surface.horizontal_vp is None or len(self.surface.horizontal_vp) == 0:
            return []

        surface_barriers = self.surface.horizontal_vp[0].inlier_lines

        # for line in self.surface.horizontal_vp[0].inlier_lines:
        #     surface_barriers.extend(line)
            
        return surface_barriers
        
            
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
            tf = TrimFinder(self.data, surface)
            surface.trim_lines = tf.solve()

        if im_logging_enabled(self.data):
            log_image(self.data, "trim.png", self.get_debug_image())


    def get_debug_image(self):

        hues = random.sample(range(0, 360), len(self.surfaces))
                    
        img = self.image.copy()

        Line.draw_all(img, self.data["lines"], color=(0,255,0), thickness=1)

        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            color = convert_color((hues[i],127,255), cv2.COLOR_HSV2RGB_FULL)

            Line.draw_all(img, surface.trim_lines, color=color, thickness=2)


        return img


