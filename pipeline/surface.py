from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import math
from scipy import ndimage
import cv2
import uuid

from .core import SurfaceType
from .geometry import Geometry
from .planegeometry import PlanarDimension
from .utils import convert_color, put_text
from .Line import line_angle_difference

class Surface():

    def __init__(self, data, index, surfaceType=None):
        self.data = data
        self.index = index
        self._uniqueId = uuid.uuid4()
        self._surfaceType = surfaceType
        self._geometry = None
        self._mask = None
        self._alteration = None

    @property
    def uniqueId(self) -> str:
        return self._uniqueId

    @property
    def surfaceType(self) -> SurfaceType:
        return self._surfaceType

    @property
    def name(self) -> Geometry:
        return "%s %d" % (self.surfaceType.name, self.index)

    @property
    def geometry(self) -> Geometry:
        return self._geometry    

    @property
    def probs(self) -> Geometry:
        return self.geometry.masks[self.index]

    @property
    def normal(self) -> float:
        return self.data["plane_normals"][self.index]

    @property
    def offset(self) -> float:
        return self.data["plane_offsets"][self.index]

    @property
    def angle(self): #from floor
        floor_normal = (0,0,-1)
        dot_product = np.dot(floor_normal, self.normal)
        result = np.arccos(dot_product)
        return 0 if np.isnan(result) else result

    @property
    def mask(self) -> ndimage:
        return self._mask

    @property
    def center(self) -> tuple:
        #todo: use 2D projection
        if self.moments is None or self.moments["m00"] == 0:
            cX = self.mask.shape[1] // 2
            cY = self.mask.shape[0] // 2
        else:
            cX = int(self.moments["m10"] / self.moments["m00"])
            cY = int(self.moments["m01"] / self.moments["m00"])
        return cX, cY

    @property
    @abstractmethod
    def dimension() -> PlanarDimension:
        pass

    def determine_surface_type(self, labels, confidence, K, angle_threshold=np.radians(20)):
        isolated = self.data["isolated"]

        mask = np.zeros(self.probs.shape, dtype="uint8")
        mask[mask < confidence] = 0
        if np.sum(mask) < 10:
            mask = self.probs

        self.isolated_probs = []
        for group in SurfaceType:
            submask = isolated[group] * mask
            submask[labels != group.index] = np.nan #exclude
            self.isolated_probs.append(submask)

        self.category_probs = np.asarray([np.nanmean(prob) for prob in self.isolated_probs])
        self.category_probs = np.nan_to_num(self.category_probs)
        self.category_counts = np.asarray([np.count_nonzero(prob[prob >= 0.1]) for prob in self.isolated_probs])

        self.best_indices = self.category_counts.argsort()[-K:][::-1]
        self.best_surface_types = list(map(lambda i: SurfaceType(i), self.best_indices))

        self._surfaceType = self.best_surface_types[0]

        #now fix incorrect classifications:
        angle_with_wall = abs(0.5 * np.pi - self.angle)
        angle_with_ceiling = abs(np.pi - self.angle)
        angle_with_floor = abs(self.angle)

        #Maybe it is being classified as ceiling when it's really wall or vice versa:
        #check the angle versus the floor normal. Walls are generally orthagonal to the floor or ceiling    
        if self.surfaceType != SurfaceType.Other:
            if angle_with_wall < angle_threshold and self.surfaceType != SurfaceType.Wall and self.surfaceType != SurfaceType.WallLike:
                self._surfaceType = SurfaceType.Wall if self.surfaceType.is_major else SurfaceType.WallLike
                self._alteration = "%d deg from wall" % int(math.degrees(angle_with_wall))
            elif angle_with_ceiling < angle_threshold and self.surfaceType != SurfaceType.Ceiling and self.surfaceType != SurfaceType.CeilingLike:
                self._surfaceType = SurfaceType.Ceiling if self.surfaceType.is_major else SurfaceType.CeilingLike
                self._alteration = "%d deg from ceil" % int(math.degrees(angle_with_ceiling))
            elif angle_with_floor < angle_threshold and self.surfaceType != SurfaceType.Floor and self.surfaceType != SurfaceType.FloorLike:
                self._surfaceType = SurfaceType.Floor if self.surfaceType.is_major else SurfaceType.FloorLike
                self._alteration = "%d deg from floor" % int(math.degrees(angle_with_floor))

        #If it is minor type, e.g. walllike or floorlike, it may need to become a major type such as wall or floor:
        # if not self.surfaceType.is_major:
        #     original_counts = self.category_counts[self.surfaceType.index]

        #     major_type = SurfaceType(self._surfaceType - 1)
        #     major_counts = self.category_counts[major_type.index]

        #     #compare the total pixels. If it's a minor type it will be smaller
        #     if major_counts > original_counts and major_type in self.best_surface_types:
        #         self._surfaceType = major_type
        #         self._alteration = "min %d->%d maj" % (original_counts, major_counts)
                
        if self._alteration is not None:
            print("Changed %d from %s to %s: %s" % (self.index, self.best_surface_types[0].name, self.surfaceType.name, self._alteration))
        


    def analyze(self, labels, confidence=0.05, K=3):

        self.determine_surface_type(labels, confidence, K)

        highest = np.max(self.probs)
        confidence = max(min(highest * 0.9, confidence), 0.05)

        self._mask = np.zeros(self.probs.shape, dtype="uint8")
        self._mask[self.probs >= confidence] = 1
        self._mask[labels != self.surfaceType] = 0

        self.contours, self.hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.moments = cv2.moments(self.contours[0]) if len(self.contours) > 0 else None

        if self.moments is None or self.moments["m00"] == 0:
            self.moments = cv2.moments(self.mask)

    def debug(self, img, color):
        cv2.drawContours(img, self.contours, -1, color)
    
        if self.center is None: 
            print("No center found for %s" % self.name)
            return

        pos = self.center
        pos = put_text(img, self.name, pos, color, size=0.5, shadow=True, highlights=True)

        if self._alteration is not None:
            pos = put_text(img, self._alteration, pos, color, size=0.33, shadow=True)

        pos = put_text(img, "%.0f deg" % np.degrees(self.angle), pos, (255, 255, 255), size=0.33, shadow=True)

        # if surface.surfaceType != surface.best_surface_types[0]:
        #     best_prob = surface.category_probs[surface.best_surface_types[0]]
        #     chosen_prob = surface.category_probs[surface.surfaceType]
        #     text = "%s %.2f to %s %.2f" % (surface.best_surface_types[0].name, best_prob, surface.surfaceType.name, chosen_prob)

        #     put_text(img, text, (position[0], position[1] + text_size[1]), (255, 255, 255), size=0.33, shadow=True) 
