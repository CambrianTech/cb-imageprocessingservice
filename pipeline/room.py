from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import cv2
import uuid
import random

from .geometry import Geometry
from .core import SurfaceType
from .surface import Surface
from .utils import convert_color, put_text, overlay_mask
from .logging import im_logging_enabled, log_image, log_segmentation_image

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

        plane_masks = np.dstack(tuple(self.masks))
        plane_masks = np.int32(np.argmax(plane_masks, -1))
        log_segmentation_image(self.data, "plane_masks", plane_masks, self.image)

        #perform initial analysis
        for surface in self.surfaces:
            surface.analyze()

        #merge heavily intersecting surfaces:
        
        def expand_into_type(surfaceType:SurfaceType):
            surfaces = self.get_surfaces(surfaceType)

            if len(surfaces) == 0:
                return

            total_mask = np.sum(np.dstack([s.mask for s in surfaces]), axis=-1)
            disputed_areas = np.zeros_like(total_mask)
            disputed_areas[total_mask > 1] = 1

            unfilled_areas = np.zeros_like(total_mask)
            where = np.where(np.logical_and(self.labels == surfaceType.index, total_mask == 0))
            unfilled_areas[where] = 1

            
            #intersecting_areas[total_mask > 1] = 1


            # for surface in surfaces:
            #     surface.mask[markers == surface.index + 1] = 1

            if im_logging_enabled(self.data):
                log_image(self.data, "room_unfilled", overlay_mask(self.image, unfilled_areas))

            


        #expand all surfaces as far as they can go within their segmentation (watershed)
        #and resolve disputes between planes as they intersect by probability (confidence):
        expand_into_type(SurfaceType.Wall)


        #find_missing_surfaces
        
        


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


