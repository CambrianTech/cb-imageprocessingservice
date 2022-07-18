from abc import abstractmethod
import numpy as np
import cv2
import uuid
import random
from termcolor import colored

from pipeline.data.surface_type import SurfaceType
from pipeline.misc.utils import resize_array, multi_filter, convert_color
from pipeline.data.logging import im_logging_enabled

indexed_fields = ["plane_parameters", "plane_normals", "plane_offsets", "plane_clusters"]

class Room():

    def __init__(self, data):
        super().__init__()
        self.data = data

        self._surfaces = {}
        self._probs = None
        self._index_mask = None
        self._normals = None
        self.vertical_vp = self.data["vertical_vp"]
        self.horizontal_vps = self.data["horizontal_vps"]
        self.barrier_lines = None

        ade_seg_c = np.dstack(tuple(self.data["isolated"]))
        self.isolated_labels = np.int32(np.argmax(ade_seg_c, -1))

        ade_seg_c = np.dstack((tuple(self.data["semantic_probs"])))
        self.semantic_labels = np.int32(np.argmax(ade_seg_c, -1))

        self._mask_intersections = {}

    @property
    def vanishing_points(self):
        vps = []
        if self.vertical_vp is not None:
            vps.append(self.vertical_vp)
        if self.horizontal_vps is not None:
            vps.extend(self.horizontal_vps)

        return vps

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

        self._surfaces[surface.index] = surface

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

        self._mask_intersections = {}

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

    @classmethod
    def surface_surface_key(cls, surface_a, surface_b):
        return str(surface_a.uniqueId) + str(surface_b.uniqueId) if surface_a.uniqueId < surface_b.uniqueId else str(surface_b.uniqueId) + str(surface_a.uniqueId)

    def surface_surface_intersection(self, surface_a, surface_b):
        key = self.surface_surface_key(surface_a, surface_b)

        if key not in self._mask_intersections:
            self._mask_intersections[key] = np.bitwise_and(surface_a.mask_expanded, surface_b.mask_expanded)

        return self._mask_intersections[key]


    def get_debug_image(self, hires=False, neighbors=False, alpha=0.666):
        if not im_logging_enabled(self.data): 
            return None

        image = self.data["image"] if hires else self.data["downscaled"]
        img_hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.surfaces))

        #overlay mask
        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            
            mask = surface.hires_mask if hires else surface.mask
        
            query = mask > 0 
            img_hsv[query] = np.mean(img_hsv[query], axis=0)
            max_value = 0.9
            if max_value > 0:
                img_hsv[:, :, 0][query] = hues[i]
                img_hsv[:, :, 1][query] = 255

        img = cv2.addWeighted(cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL), alpha, image, (1.0 - alpha), 0.0)

        #let surface do its debug
        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            hue = hues[i]
            color = convert_color((hue, 255, 255), cv2.COLOR_HSV2RGB_FULL)
            surface.debug(img, color)

        return img