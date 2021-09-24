from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import cv2
import uuid
import random

from .geometry import Geometry
from .core import SurfaceType
from .surface import Surface
from .utils import convert_color, put_text, overlay_mask
from .logging import im_logging_enabled, log_image

#python info on object oriented methods and properties
#https://stackoverflow.com/questions/2736255/abstract-attributes-in-python
class Room(Geometry):

    def __init__(self, data):
        super().__init__(data)
        
        # self.vert_indices = self.data["dimensions"][Dimension.Vertical].indices
        # self.horiz_indices = self.data["dimensions"][Dimension.Vertical].indices

    @property
    def ceilings(self):
        return self.get_surfaces(surfaceType=SurfaceType.Ceiling)

    @property
    def walls(self):
        return self.get_surfaces(surfaceType=SurfaceType.Wall)

    @property
    def floors(self):
        return self.get_surfaces(surfaceType=SurfaceType.Floor)

    def analyze(self, labels):
        self.labels = labels

        #perform initial analysis
        for surface in self.surfaces:
            surface.analyze()

        #merge heavily intersecting surfaces:
        total_mask = np.sum(np.dstack([s.mask for s in self.surfaces]), axis=-1)
        intersecting_areas = np.zeros_like(total_mask)
        intersecting_areas[total_mask > 1] = 1

        
        #expand all surfaces as far as they can go within their segmentation (watershed)
        #and resolve disputes between planes as they intersect by probability (confidence):
        markers = np.zeros(self.image.shape[:2], dtype=np.int32)
        for surface in self.surfaces:
            markers[surface.mask > 0] = (surface.index + 1)

        markers[intersecting_areas > 0] = 0
        #find_missing_surfaces


        if im_logging_enabled(self.data):
            log_image(self.data, "room_markers", markers * (255 / len(self.surfaces)))
            log_image(self.data, "room_intersect", overlay_mask(self.image, intersecting_areas))
            log_image(self.data, "room_total_mask", overlay_mask(self.image, total_mask * 255 / np.max(total_mask)))
        
        


    def get_debug_image(self, masked=True):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.surfaces))

        #overlay probs
        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]

            mask = surface.mask > 0 if masked else surface.probs >= 0.05

            img_hsv[:, :, 0][mask] = hues[i]
            img_hsv[:, :, 1][mask] = 255 * np.power(surface.probs[mask], 0.5)
                    
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)

        #let surface do its debug
        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            hue = hues[i]
            color = convert_color((hue, 255, 255), cv2.COLOR_HSV2RGB_FULL)

            surface.debug(img, color)
                

        return img


