from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import cv2
import uuid

from .utils import resize_array, multi_filter
from .Line import Line
from .logging import log_segmentation_image

class Geometry():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self._surfaces = {}
        self.image = self.data["downscaled"]
        planes_data = self.data["planes"]
        shape = (self.image.shape[1], self.image.shape[0])
        self.probs = resize_array(planes_data["masks"], shape)

        self.masks = np.dstack(tuple(self.probs))
        self.masks = np.int32(np.argmax(self.masks, -1))

        ade_seg_c = np.dstack(tuple(self.data["isolated"]))
        self.labels = np.int32(np.argmax(ade_seg_c, -1))

        self.lines = self.data["lines"]

    def add_surface(self, surface):
        surface._geometry = self
        self._surfaces[surface.uniqueId] = surface

    def get_surfaces(self, surfaceType=None, dimension=None):
        filters = [lambda surface: not surface.invalidated]
        
        if surfaceType is not None:
            filters.append(lambda surface: surface.surfaceType == surfaceType)

        if dimension is not None:
            filters.append(lambda surface: surface.dimension == dimension)

        return self._surfaces.values() if len(filters) is None else list(multi_filter(filters, self._surfaces.values()))

    @property
    def surfaces(self):
        return self.get_surfaces()

    @abstractmethod
    def get_debug_image(self, confidence=0.05):
        pass