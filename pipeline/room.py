from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from scipy import ndimage
import cv2
from enum import Enum
import uuid
import random
from skimage.morphology import skeletonize, thin

from .utils import multi_filter, resize_array, overlay_mask, convert_color, put_text
from .planegeometry import PlanarDimension
from .extractsurfaces import Groupings

#python info on object oriented methods and properties
#https://stackoverflow.com/questions/2736255/abstract-attributes-in-python
class Geometry():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self._surfaces = {}
        self.image = self.data["downscaled"]
        planes_data = self.data["planes"]
        shape = (self.image.shape[1], self.image.shape[0])
        self.masks = resize_array(planes_data["masks"], shape)

    def add_surface(self, surface):
        surface._geometry = self
        self._surfaces[surface.uniqueId] = surface

    def get_surfaces(self, surfaceType=None, dimension=None):
        filters = []
        
        if surfaceType is not None:
            filters.append(lambda surface: surface.surfaceType == surfaceType)

        if dimension is not None:
            filters.append(lambda surface: surface.dimension == dimension)

        return self._surfaces.values() if len(filters) is None else list(multi_filter(filters, self._surfaces.values()))

    @property
    def surfaces(self):
        return self.get_surfaces()

class Surface():

    def __init__(self, data, index, surfaceType=None):
        self.data = data
        self.index = index
        self._uniqueId = uuid.uuid4()
        self._surfaceType = surfaceType
        self._geometry = None
        self._mask = None

    @property
    def surfaceType(self) -> Groupings:
        return self._surfaceType

    @property
    def uniqueId(self) -> str:
        return self._uniqueId

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def probs(self) -> Geometry:
        return self.geometry.masks[self.index]

    @property
    def center(self) -> tuple:
        if self.moments is None or self.moments["m00"] == 0:
            return None

        cX = int(self.moments["m10"] / self.moments["m00"])
        cY = int(self.moments["m01"] / self.moments["m00"])
        return cX, cY

    @property
    @abstractmethod
    def dimension() -> PlanarDimension:
        pass

    def analyze(self, labels, confidence, bounds_confidence=0.05, K=3):

        isolated = self.data["isolated"]

        highest = np.max(self.probs)
        confidence = max(min(highest * 0.95, confidence), 0.05)

        bounds_confidence = max(min(highest * 0.95, bounds_confidence), 0.05)

        mask = np.zeros(self.probs.shape, dtype="uint8")
        mask[mask < confidence] = 0
        if np.sum(mask) < 10:
            mask = self.probs


        #something simpler like vectorize?
        self.isolated_probs = []
        for group in Groupings:
            submask = isolated[group] * mask
            submask[labels != group.index] = np.nan
            self.isolated_probs.append(submask)

        self.category_probs = np.asarray([np.nanmean(i) for i in self.isolated_probs])
        self.category_probs = np.nan_to_num(self.category_probs)
        self.best_indices = self.category_probs.argsort()[-K:][::-1]

        self.best_surface_types = list(map(lambda i: Groupings(i), self.best_indices))

        self._surfaceType = self.best_surface_types[0]

        #todo: check angles and other things to verify that ceilings are the right angle to be that 
        # and walls are vertical, floors horizontal but opposite ceilings:

        #todo: as of right now, this is just used for visualization, so protect inside debug section:
        mask = np.zeros(self.probs.shape, dtype="uint8")
        mask[self.probs > bounds_confidence] = 1
        self.contours, self.hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.moments = cv2.moments(self.contours[0]) if len(self.contours) > 0 else None

        if self.moments is None or self.moments["m00"] == 0:
            self.moments = cv2.moments(mask)

        #print("Category", self.category)

        # best_match = Groupings.Unknown
        # floor_probs = np.sum(isolated[Groupings.Floor] * self.mask)
        # wall_probs = np.sum(isolated[Groupings.Wall] * self.mask)
        # ceiling_probs = np.sum(isolated[Groupings.Ceiling] * self.mask)

class Room(Geometry):

    def __init__(self, data):
        super().__init__(data)
        
        # self.vert_indices = self.data["dimensions"][Dimension.Vertical].indices
        # self.horiz_indices = self.data["dimensions"][Dimension.Vertical].indices

    @property
    def ceilings(self):
        return self.get_surfaces(surfaceType=Groupings.Ceiling)

    @property
    def walls(self):
        return self.get_surfaces(surfaceType=Groupings.Wall)

    @property
    def floors(self):
        return self.get_surfaces(surfaceType=Groupings.Floor)

    def analyze(self, labels, confidence=0.3):
        
        for surface in self.surfaces:
            surface.analyze(labels, confidence)

        # isolated[Groupings.Floor]
        # isolated[Groupings.Wall]
        # isolated[Groupings.Ceiling]

    def get_debug_image(self, confidence=0.05):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.surfaces))

        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]

            if surface.surfaceType != Groupings.Other:
                img_hsv[:, :, 0][surface.probs >= confidence] = hues[i]
                img_hsv[:, :, 1][surface.probs >= confidence] = 255 * np.power(surface.probs[surface.probs > confidence], 0.5)

    
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)

        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            hue = hues[i]

            if surface.surfaceType != Groupings.Other:
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


