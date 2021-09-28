from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import cv2
import uuid

from .utils import resize_array, multi_filter

class Geometry():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self._surfaces = {}
        self.image = self.data["downscaled"]
        planes_data = self.data["planes"]
        shape = (self.image.shape[1], self.image.shape[0])

        self.probs = resize_array(planes_data["masks"], shape)
        self.index_mask = np.dstack(tuple(self.probs))
        self.index_mask = np.int32(np.argmax(self.index_mask, -1))

        ade_seg_c = np.dstack(tuple(self.data["isolated"]))
        self.labels = np.int32(np.argmax(ade_seg_c, -1))

    def add_surface(self, surface):
        if surface.index < 0:
            #get next index, expand everything
            surface.index = len(self.data["planes"])
            #self.probs.add_row

        surface._geometry = self
        self._surfaces[surface.index] = surface

    def refresh_surfaces(self):        
        self._surfaces = dict(filter(lambda kv:not kv[1].invalidated, self._surfaces.items()))

    def get_surfaces(self, surfaceType=None, dimension=None):

        self.refresh_surfaces()

        filters = []
        
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