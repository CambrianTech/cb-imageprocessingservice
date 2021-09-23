from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
from scipy import ndimage
import cv2
import uuid

from .core import SurfaceType
from .geometry import Geometry
from .planegeometry import PlanarDimension

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

        self.category_probs = np.asarray([np.nanmean(i) for i in self.isolated_probs])
        self.category_probs = np.nan_to_num(self.category_probs)
        self.best_indices = self.category_probs.argsort()[-K:][::-1]

        self.best_surface_types = list(map(lambda i: SurfaceType(i), self.best_indices))

        self._surfaceType = self.best_surface_types[0]

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
