from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from scipy import ndimage
import cv2
import uuid

from .core import SurfaceType
from .geometry import Geometry
from .planegeometry import PlanarDimension
from .utils import convert_color, put_text

class Surface():

    def __init__(self, data, index, surfaceType=None):
        self.data = data
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
    def mask(self) -> ndimage:
        return self._mask

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

    def determine_surface_type(self, labels, confidence, K):
        isolated = self.data["isolated"]

        mask = np.zeros(self.probs.shape, dtype="uint8")
        mask[mask < confidence] = 0
        if np.sum(mask) < 10:
            mask = self.probs

        self.isolated_probs = []
        for group in SurfaceType:
            submask = isolated[group] * mask
            submask[labels != group.index] = np.nan
            self.isolated_probs.append(submask)

        self.category_probs = np.asarray([np.nanmean(prob) for prob in self.isolated_probs])
        self.category_probs = np.nan_to_num(self.category_probs)

        self.category_counts = [np.count_nonzero(prob[prob >= 0.1]) for prob in self.isolated_probs]

        self.best_indices = self.category_probs.argsort()[-K:][::-1]
        self.best_surface_types = list(map(lambda i: SurfaceType(i), self.best_indices))

        self._surfaceType = self.best_surface_types[0]
        self._altered = False

        if self._surfaceType % 2 == 1: #is minor type, aka walllike, floorlike, ceilinglike
            original_type = self._surfaceType
            original_counts = self.category_counts[original_type.index]

            major_type = SurfaceType(self._surfaceType - 1)
            major_counts = self.category_counts[major_type.index]

            if major_counts > original_counts and major_type in self.best_surface_types:
                print("Switchin type from %s to %s: %d->%d" % (original_type.name, major_type.name, original_counts, major_counts))
                self._surfaceType = major_type
                self._altered = True

        #todo: check angles and other things to verify that ceilings are the right angle to be that 
        # and walls are vertical, floors horizontal but opposite ceilings:

    def analyze(self, labels, confidence, bounds_confidence=0.05, K=3):

        self.determine_surface_type(labels, confidence, K)

        highest = np.max(self.probs)
        confidence = max(min(highest * 0.95, confidence), 0.05)
        bounds_confidence = max(min(highest * 0.95, bounds_confidence), 0.05)

        self._mask = np.zeros(self.probs.shape, dtype="uint8")
        self._mask[self.probs > bounds_confidence] = 1
        self._mask[labels != self.surfaceType] = 0

        self.contours, self.hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.moments = cv2.moments(self.contours[0]) if len(self.contours) > 0 else None

        if self.moments is None or self.moments["m00"] == 0:
            self.moments = cv2.moments(self.mask)

    def debug(self, img, color):
        cv2.drawContours(img, self.contours, -1, color)
    
        if self.center is None: return

        text_size, position = put_text(img, self.surfaceType.name, self.center, color, size=0.5, shadow=True, highlights=True)

        if self._altered:
            text = "%d" % (self.category_counts[self.surfaceType.index])
            put_text(img, text, (position[0], position[1] + text_size[1]), (255, 255, 255), size=0.33, shadow=True)

        # if surface.surfaceType != surface.best_surface_types[0]:
        #     best_prob = surface.category_probs[surface.best_surface_types[0]]
        #     chosen_prob = surface.category_probs[surface.surfaceType]
        #     text = "%s %.2f to %s %.2f" % (surface.best_surface_types[0].name, best_prob, surface.surfaceType.name, chosen_prob)

        #     put_text(img, text, (position[0], position[1] + text_size[1]), (255, 255, 255), size=0.33, shadow=True) 
