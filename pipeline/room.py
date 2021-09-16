from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from scipy import ndimage
import cv2
from enum import Enum
import uuid
import random

from .utils import multi_filter, resize_array, overlay_mask
from .planegeometry import PlanarDimension

#python info on object oriented methods and properties
#https://stackoverflow.com/questions/2736255/abstract-attributes-in-python

class SurfaceType(Enum):
    Unknown = "unknown"
    Horizontal = "horizontal"
    Vertical = "vertical"
    Floor = "floor"
    Wall = "wall"
    Ceiling = "ceiling"

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

    def __init__(self, data, index, surfaceType=SurfaceType.Unknown):
        self._data = data
        self.index = index
        self._uniqueId = uuid.uuid4()
        self._surfaceType = surfaceType
        self._geometry = None
        self._mask = None

    @property
    def surfaceType(self) -> SurfaceType:
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
    def mask(self, confidence=0.05):
        if self._mask is None:
            mask = self.probs.copy()
            mask[mask < confidence] = 0
            mask[mask > 0] = 255
            self._mask = np.uint8(mask)
        return self._mask

    @property
    @abstractmethod
    def dimension() -> PlanarDimension:
        pass

    def analyze(self):
        print("Analyzing surface")
        self.contours = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

class HorizontalSurface(Surface):

    def __init__(self, data, index, surfaceType=SurfaceType.Horizontal):
        super().__init__(data, index, surfaceType)

    @property
    @abstractmethod
    def dimension() -> PlanarDimension:
        return PlanarDimension.Horizontal

class VerticalSurface(Surface):

    def __init__(self, data, index, surfaceType=SurfaceType.Horizontal):
        super().__init__(data, index, surfaceType)

    @property
    @abstractmethod
    def dimension() -> PlanarDimension:
        return PlanarDimension.Vertical

class Ceiling(HorizontalSurface):

    def __init__(self, data, index, surfaceType=SurfaceType.Ceiling):
        super().__init__(data, index, surfaceType)


class Floor(HorizontalSurface):

    def __init__(self, data, index, surfaceType=SurfaceType.Floor):
        super().__init__(data, index, surfaceType)

class Wall(VerticalSurface):

    def __init__(self, data, index, surfaceType=SurfaceType.Wall):
        super().__init__(data, index, surfaceType)



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

    def analyze(self):
        for surface in self.surfaces:
            surface.analyze()

    def get_debug_image(self, confidence=0.07):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV) #range 0-180
        for surface in self.surfaces:
            hue = random.randint(0,180)
            img_hsv[:, :, 0][surface.probs >= confidence] = hue 
            img_hsv[:, :, 1][surface.probs >= confidence] = 255 * np.power(surface.probs[surface.probs > confidence], 0.5)

        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)

        return img


