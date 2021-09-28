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
        self._probs = None
        self._index_mask = None

        ade_seg_c = np.dstack(tuple(self.data["isolated"]))
        self.labels = np.int32(np.argmax(ade_seg_c, -1))

    def add_surface(self, surface):
        if surface.index < 0:
            #get next index, expand everything
            surface.index = len(self.data["planes"])
            surface_mask = self.data["planes"]["masks"][surface.cloned_from].copy()
            self.data["planes"]["masks"] = np.append(self.data["planes"]["masks"], [surface_mask], axis=0)
            self.invalidate()

        surface._geometry = self
        self._surfaces[surface.index] = surface

    @property
    def image(self):
        return self.data["downscaled"]

    @property
    def surfaces(self):
        return self.get_surfaces()

    @property
    def probs(self):
        if self._probs is None:
            shape = (self.image.shape[1], self.image.shape[0])
            self._probs = resize_array(self.data["planes"]["masks"], shape)

        return self._probs

    @property
    def index_mask(self):
        if self._index_mask is None:
            self._index_mask = np.dstack(tuple(self.probs))
            self._index_mask = np.int32(np.argmax(self._index_mask, -1))
        return self._index_mask

    @abstractmethod
    def get_debug_image(self, confidence=0.05):
        pass

    def invalidate(self):
        self._probs = None
        self._index_mask = None

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