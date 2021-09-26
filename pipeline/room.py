from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import cv2
import uuid
import random
from skimage.segmentation import watershed

from .geometry import Geometry
from .core import SurfaceType
from .surface import Surface
from .utils import convert_color, put_text, overlay_mask
from .logging import im_logging_enabled, log_image, log_segmentation_image

from termcolor import colored

#python info on object oriented methods and properties
#https://stackoverflow.com/questions/2736255/abstract-attributes-in-python
class Room(Geometry):

    @property
    def ceilings(self):
        return self.get_surfaces(surfaceType=SurfaceType.Ceiling)

    @property
    def walls(self):
        return self.get_surfaces(surfaceType=SurfaceType.Wall)

    @property
    def floors(self):
        return self.get_surfaces(surfaceType=SurfaceType.Floor)

    def analyze(self):
        
        #perform initial analysis
        num_before = len(self.surfaces)
        for surface in self.surfaces:
            surface.analyze()

        num_after = len(self.surfaces)
        if num_after != num_before:
            print(colored("Surfaces reduced from %d to %d" % (num_before, num_after), 'red'))
        
        def expand_into_type(surfaceType:SurfaceType):
            surfaces = self.get_surfaces(surfaceType)

            if len(surfaces) == 0:
                return

            total_mask = np.sum(np.dstack([s.mask for s in surfaces]), axis=-1)
            disputed_areas = np.zeros_like(total_mask)
            disputed_areas[total_mask > 1] = 1

            markers = np.zeros(total_mask.shape, dtype=np.int32)
            
            for index in range(len(surfaces)):
                surface = surfaces[index]
                markers[surface.mask > 0] = index + 1

            markers[disputed_areas > 0] = 0
            markers[self.labels != surfaceType.index] = 255

            if im_logging_enabled(self.data):
                log_image(self.data, "room_%s_markers" % surfaceType.name, markers * 20)

            #perform watershed:            
            markers = watershed(self.image, markers)
            markers[markers<0] = 0

            if im_logging_enabled(self.data):
                log_image(self.data, "room_%s_watershed" % surfaceType.name, markers * 20)


        #expand all surfaces as far as they can go within their segmentation (watershed)
        #and resolve disputes between planes as they intersect by probability (confidence):
        expand_into_type(SurfaceType.Wall)
        #expand_into_type(SurfaceType.Floor)

        #find_missing_surfaces
        
    def get_debug_image(self):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.surfaces))

        #overlay probs
        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            mask = surface.mask > 0

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


