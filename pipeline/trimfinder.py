import numpy as np

import cv2
import random
import time
import math

from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .logging import log_image, im_logging_enabled
from .utils import convert_color


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

        if im_logging_enabled(self.data):
            log_image(self.data, "trim.png", self.get_debug_image())


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


        return img


