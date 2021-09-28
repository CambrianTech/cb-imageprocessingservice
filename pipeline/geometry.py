from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import cv2
import uuid
from termcolor import colored

from .utils import resize_array, multi_filter

indexed_fields = ["plane_parameters", "plane_normals", "plane_offsets", "plane_clusters"]

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
            #get next index, expand everything, 
            #todo: store inside surface
            surface.index = self.data["planes"]["masks"].shape[0]

            #append a copy of the index mask
            surface_mask = self.data["planes"]["masks"][surface.cloned_from].copy() #check for intersect?
            self.data["planes"]["masks"] = np.append(self.data["planes"]["masks"], [surface_mask], axis=0)

            for field in indexed_fields:
                self.data[field] = np.append(self.data[field], [self.data[field][surface.cloned_from].copy()], axis=0)

            self.invalidate()

        surface._geometry = self
        self._surfaces[surface.index] = surface

    def remove_surface(self, surface):
        surface.destroyed = True
        #remove other stuff?

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
        num_before = len(self._surfaces)
        self._surfaces = dict(filter(lambda kv:not kv[1].destroyed, self._surfaces.items()))

        num_after = len(self._surfaces)
        if num_after < num_before:
            print(colored("Surfaces reduced from %d to %d" % (num_before, num_after), 'red'))
            #cleanup:

    def get_surfaces(self, surfaceType=None, dimension=None):

        self.refresh_surfaces()

        filters = []
        
        if surfaceType is not None:
            filters.append(lambda surface: surface.surfaceType == surfaceType)

        if dimension is not None:
            filters.append(lambda surface: surface.dimension == dimension)

        return self._surfaces.values() if len(filters) is None else list(multi_filter(filters, self._surfaces.values()))