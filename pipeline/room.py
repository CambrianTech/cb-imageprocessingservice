from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from scipy import ndimage
import cv2
from enum import Enum
import uuid

from .utils import multi_filter
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

class Surface():

    def __init__(self, data, surfaceType=SurfaceType.Unknown):
        self._data = data
        self._uniqueId = uuid.uuid4()
        self._surfaceType = surfaceType

    @property
    def surfaceType(self) -> SurfaceType:
        return self._surfaceType

    @property
    def uniqueId(self) -> str:
        return self._uniqueId

    @property
    @abstractmethod
    def dimension() -> PlanarDimension:
        pass

class HorizontalSurface(Surface):

    def __init__(self, data, surfaceType=SurfaceType.Horizontal):
        super().__init__(data, surfaceType)

    @property
    @abstractmethod
    def dimension() -> PlanarDimension:
        return PlanarDimension.Horizontal

class VerticalSurface(Surface):

    def __init__(self, data, surfaceType=SurfaceType.Horizontal):
        super().__init__(data, surfaceType)

    @property
    @abstractmethod
    def dimension() -> PlanarDimension:
        return PlanarDimension.Vertical

class Ceiling(HorizontalSurface):

    def __init__(self, data, surfaceType=SurfaceType.Ceiling):
        super().__init__(data, surfaceType)


class Floor(HorizontalSurface):

    def __init__(self, data, surfaceType=SurfaceType.Floor):
        super().__init__(data, surfaceType)

class Wall(VerticalSurface):

    def __init__(self, data, surfaceType=SurfaceType.Wall):
        super().__init__(data, surfaceType)


class Geometry():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.surfaces = {}

    def add_surface(self, surface):
        self.surfaces[surface.uniqueId] = surface

    def get_surfaces(self, surfaceType=None, dimension=None):
        filters = []
        
        if surfaceType is not None:
            filters.append(lambda surface: surface.surfaceType == surfaceType)

        if dimension is not None:
            filters.append(lambda surface: surface.dimension == dimension)

        return self.surfaces if len(filters) is None else list(multi_filter(filters, self.surfaces))


class Room(Geometry):

    def __init__(self, data):
        super().__init__(data)

    def ceilings(self):
        return self.get_surfaces(surfaceType=SurfaceType.Ceiling)

    def walls(self):
        return self.get_surfaces(surfaceType=SurfaceType.Wall)

    def floors(self):
        return self.get_surfaces(surfaceType=SurfaceType.Floor)


