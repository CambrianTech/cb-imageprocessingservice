from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import cv2
import uuid
import random

from .geometry import Geometry
from .core import SurfaceType
from .surface import Surface
from .utils import convert_color, put_text

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

    def analyze(self, labels, confidence=0.3):
        
        for surface in self.surfaces:
            surface.analyze(labels, confidence)

    def get_debug_image(self, confidence=0.05):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.surfaces))

        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]

            if surface.surfaceType != SurfaceType.Other:
                img_hsv[:, :, 0][surface.probs >= confidence] = hues[i]
                img_hsv[:, :, 1][surface.probs >= confidence] = 255 * np.power(surface.probs[surface.probs > confidence], 0.5)

    
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)

        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            hue = hues[i]

            if surface.surfaceType != SurfaceType.Other:
                color = convert_color((hue, 255, 255), cv2.COLOR_HSV2RGB_FULL)
                cv2.drawContours(img, surface.contours, -1, color)

                if surface.center is None:
                    continue

                text_size, position = put_text(img, surface.surfaceType.name, surface.center, color, size=0.5, shadow=True, highlights=True)

                if surface.surfaceType != surface.best_surface_types[0]:
                    best_prob = surface.category_probs[surface.best_surface_types[0]]
                    chosen_prob = surface.category_probs[surface.surfaceType]
                    text = "%s %.2f to %s %.2f" % (surface.best_surface_types[0].name, best_prob, surface.surfaceType.name, chosen_prob)

                    put_text(img, text, (position[0], position[1] + text_size[1]), (255, 255, 255), size=0.33, shadow=True)                
                

        return img


