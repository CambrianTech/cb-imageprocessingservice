from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import cv2
import uuid
from termcolor import colored

from pipeline.data.surface_type import SurfaceType
from pipeline.misc.utils import resize_array, multi_filter

indexed_fields = ["plane_parameters", "plane_normals", "plane_offsets", "plane_clusters"]

class PlanarGroup():
    def __init__(self, surface):
        super().__init__()
        self.uniqueId = uuid.uuid4()
        self.add_surface(surface)

    def add_surface(self, surface):
        surface.planar_group = self

    @property
    def normal(self) -> tuple:
        #todo: maybe composite such as mode or mean
        return self.surfaces[0].normal

    @property
    def offset(self) -> float:
        #todo: maybe composite such as mode, mean, max, or min
        return self.surfaces[0].offset

class Geometry():

    def __init__(self, data):
        super().__init__()
        self.data = data

        self._surfaces = {}
        self._probs = None
        self._index_mask = None
        self._normals = None

        ade_seg_c = np.dstack(tuple(self.data["isolated"]))
        self.isolated_labels = np.int32(np.argmax(ade_seg_c, -1))

        ade_seg_c = np.dstack((tuple(self.data["output"])))
        self.semantic_labels = np.int32(np.argmax(ade_seg_c, -1))
        self.planar_groups = []

    def add_surface(self, surface):

        surface.geometry = self
        
        if not surface.was_added:
            #get next index, expand everything, 
            #todo: store inside surface
            surface.index = self.data["planes"]["masks"].shape[0]

            #append a copy of the index mask
            surface_mask = self.data["planes"]["masks"][surface.cloned_from].copy() #check for intersect?
            self.data["planes"]["masks"] = np.append(self.data["planes"]["masks"], [surface_mask], axis=0)

            for field in indexed_fields:
                self.data[field] = np.append(self.data[field], [self.data[field][surface.cloned_from].copy()], axis=0)

            self.invalidate()

        self._surfaces[surface.index] = surface

        surface.on_added()

    def remove_surface(self, surface):
        surface.destroyed = True
        #remove other stuff?

    @property
    def image(self):
        return self.data["downscaled"]


    @property
    def normals(self):
        if self._normals is None:
            self._normals = cv2.resize(self.data["normals"], (self.image.shape[1], self.image.shape[0]))
        return self._normals

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

    def surface_at_point(self, point):

        #todo:just use index_mask
        for surface in self.surfaces:
            if point[0] < surface.mask.shape[1] and point[1] < surface.mask.shape[0]:
                if surface.mask[int(point[1]), int(point[0])] > 0:
                    return surface

        return None

    def get_surfaces(self, surfaceTypes=None, labels=None, dimension=None, point=None):

        self.refresh_surfaces()

        filters = []
        
        if surfaceTypes is not None:
            filters.append(lambda surface: surface.surfaceType in surfaceTypes)

        if labels is not None:
            filters.append(lambda surface: surface.bestLabel in labels)

        if dimension is not None:
            filters.append(lambda surface: surface.dimension == dimension)

        return self._surfaces.values() if len(filters) is None else list(multi_filter(filters, self._surfaces.values()))        