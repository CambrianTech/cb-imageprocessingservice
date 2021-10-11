import abc
from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import math
from scipy import ndimage
import cv2
import uuid
from copy import copy, deepcopy
from termcolor import colored

from .core import SurfaceType
from .geometry import Geometry
from .utils import convert_color, put_text
from .Line import line_angle_difference, Line
from .ade20k import ADE20K

class Surface():

    def __init__(self, data, index=None, surfaceType=None):
        self.data = data
        self._index = index
        self.uniqueId = uuid.uuid4()
        self.surfaceType = surfaceType
        self.geometry = None
        self._mask = None
        self._alteration = None
        self.destroyed = False
        self._cloned_from = -1
        self._plane_mask = None
        self._contours = None
        self._normals_color = None
        self._lines = None

        self._semantic_labels = None

        self._normal = None
        self._offset = None

    @property
    def secondaryType(self) -> SurfaceType:
        return next(filter(lambda t: t != self.surfaceType, self.best_surface_types))

    @property
    def name(self) -> str:
        return "%s %d" % ("?" if self.bestLabel is None else self.bestLabel.name, self.index)

    @property #protected or private
    def index(self) -> int:
        return self._index if self.was_added else self._cloned_from

    @index.setter
    def index(self, value:int):
        self._index = value

    @property
    def was_added(self) -> bool:
        return self._index is not None

    def on_added(self):
        if self._normal is not None:
            self.normal = self._normal
            self._normal = None

        if self._offset is not None:
            self.offset = self._offset
            self._offset = None

    @property
    def probs(self) -> Geometry:
        return self.geometry.probs[self.index]

    @property
    def plane_mask(self) -> Geometry:
        if self._plane_mask is None:
            self._plane_mask = np.zeros_like(self.geometry.index_mask)
            self._plane_mask[self.geometry.index_mask == self.index] = 1
        return self._plane_mask

    @property
    def normal(self) -> tuple:
        return self.data["plane_normals"][self.index]

    @normal.setter
    def normal(self, value:tuple):
        if self.was_added:
            self.data["plane_normals"][self.index] = value
        else:
            self._normal = value

    @property
    def offset(self) -> float:
        return self.data["plane_offsets"][self.index]

    @offset.setter
    def offset(self, value:float):
        if self.was_added:
            self.data["plane_offsets"][self.index] = value
        else:
            self._offset = value

    @property
    def normals_color(self) -> tuple:
        if self._normals_color is None:
            self._normals_color = np.mean(self.data["normals"], axis=(0, 1))
        return self._normals_color

    @property
    def lines(self) -> list:

        if self._lines is None:
            self._lines = []
            for i in range(len(self.data["lines"])):
                line = self.data["lines"][i]
                for contour in self.contours:
                    area = cv2.contourArea(contour)
                    padding = math.sqrt(area) / 10

                    def is_inside(point):
                        #positive (inside), negative (outside), or zero (on an edge)
                        dist = cv2.pointPolygonTest(contour, point, True)
                        return dist >= 0 or abs(dist) <= padding

                    midpoint_a = ((line.point_a[0] + line.midpoint[0]) / 2, (line.point_a[1] + line.midpoint[1]) / 2)
                    midpoint_b = ((line.point_b[0] + line.midpoint[0]) / 2, (line.point_b[1] + line.midpoint[1]) / 2)

                    if is_inside(line.midpoint) and (is_inside(midpoint_a) or is_inside(midpoint_b)):
                        self._lines.append(line)
                        break
                            
        return self._lines

    @property
    def angle(self): #from floor
        floor_normal = (0,0,-1)
        dot_product = np.dot(floor_normal, self.normal)
        result = np.arccos(dot_product)
        return 0 if np.isnan(result) else result

    @property
    def mask(self) -> ndimage:
        if self._mask is None:
            self.set_mask(self.get_surface_mask(self.surfaceType, self.confidence))
        return self._mask

    def set_mask(self, mask):
        self._mask = mask
        self.mask_changed()

    def mask_changed(self):
        self._contours = None
        self._semantic_labels = None
        self._normals_color = None

        self.geometry.invalidate()

    @property
    def contours(self) -> ndimage:
        if self._contours is None:
            self._contours, self.hierarchy = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            self.moments = cv2.moments(self._contours[0]) if len(self._contours) > 0 else None

            if self.moments is None or self.moments["m00"] == 0:
                self.moments = cv2.moments(self.mask)

        return self._contours

    @property
    def semantic_labels(self) -> ndimage:
        if self._semantic_labels is None:
            segments, counts = np.unique(self.geometry.semantic_labels[self.mask > 0], return_counts=True)
            segmentList = zip(segments.tolist(), counts.tolist())
            self._semantic_labels = sorted(segmentList, key=lambda x:-x[1])

        return self._semantic_labels

    @property
    def bestLabel(self) -> tuple:
        if len(self.semantic_labels):
            return ADE20K(self.semantic_labels[0][0] + 1)
        return None

    def clone(self):
        new_surface = copy(self)
        new_surface.uniqueId = uuid.uuid4()
        new_surface._index = None #nothing till added to room/geometry
        new_surface._cloned_from = self.index
        new_surface.mask_changed()
        
        return new_surface

    @property
    def cloned_from(self) -> int:
        return self._cloned_from

    def merge(self, surface):
        print(colored("Merge %s with %s" % (self.name, surface.name), 'magenta'))
        self._mask[surface.mask > 0] = 1

        #average the normals? Take one over the other by area? Do what where?

        self.mask_changed()

        surface.destroy()

    def destroy(self):
        self.geometry.remove_surface(self) 
        #Important! do not add code here, add inside remove_surface, and call public methods on this object

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

    def get_surface_mask(self, label:SurfaceType, confidence):
        mask = np.zeros(self.probs.shape, dtype="uint8")
        mask[self.probs >= confidence] = 1
        mask[self.geometry.isolated_labels != label.index] = 0
        return mask

    def determine_surface_type(self, K, angle_threshold=np.radians(20)):

        isolated = self.data["isolated"]

        prob_mask = self.probs.copy()
        prob_mask[self.plane_mask == 0] = 0
        if np.sum(prob_mask) < 10:
            prob_mask = self.probs

        self.isolated_probs = []
        for group in SurfaceType:
            submask = isolated[group] * prob_mask
            submask[self.geometry.isolated_labels != group.index] = np.nan #exclude
            self.isolated_probs.append(submask)

        self.category_probs = np.asarray([np.nanmean(prob) for prob in self.isolated_probs])
        self.category_probs = np.nan_to_num(self.category_probs)
        self.category_counts = np.asarray([np.count_nonzero(prob[prob >= self.confidence]) for prob in self.isolated_probs])

        self.best_indices = self.category_counts.argsort()[-K:][::-1]
        self.best_surface_types = list(map(lambda i: SurfaceType(i), self.best_indices))

        self.surfaceType = self.best_surface_types[0]

        #now fix incorrect classifications:
        angle_with_wall = abs(0.5 * np.pi - self.angle)
        angle_with_ceiling = abs(np.pi - self.angle)
        angle_with_floor = abs(self.angle)

        #Maybe it is being classified as ceiling when it's really wall or vice versa:
        #check the angle versus the floor normal. Walls are generally orthagonal to the floor or ceiling    
        if self.surfaceType != SurfaceType.Other:
            was_ceiling = self.surfaceType == SurfaceType.Ceiling
            if angle_with_wall < angle_threshold and self.surfaceType != SurfaceType.Wall and self.surfaceType != SurfaceType.WallLike:
                self.surfaceType = SurfaceType.Wall if self.surfaceType.is_major else SurfaceType.WallLike
                self._alteration = "%d deg from wall" % int(math.degrees(angle_with_wall))
            elif angle_with_ceiling < angle_threshold and self.surfaceType != SurfaceType.Ceiling and self.surfaceType != SurfaceType.CeilingLike:
                self.surfaceType = SurfaceType.Ceiling if self.surfaceType.is_major else SurfaceType.CeilingLike
                self._alteration = "%d deg from ceil" % int(math.degrees(angle_with_ceiling))
            elif angle_with_floor < angle_threshold and self.surfaceType != SurfaceType.Floor and self.surfaceType != SurfaceType.FloorLike:
                self.surfaceType = SurfaceType.Floor if self.surfaceType.is_major else SurfaceType.FloorLike
                self._alteration = "%d deg from floor" % int(math.degrees(angle_with_floor))

            if was_ceiling and self.surfaceType != SurfaceType.Ceiling:
                self.destroy() #too problematic

        #If it is minor type, e.g. walllike or floorlike, it may need to become a major type such as wall or floor:
        if not self.surfaceType.is_major:
            minor_counts = self.category_counts[self.surfaceType.index]

            major_type = SurfaceType(self.surfaceType - 1)
            major_counts = self.category_counts[major_type.index]

            primary_prob = self.category_probs[self.surfaceType]
            secondary_prob = self.category_probs[self.secondaryType]
            sp_ratio = (secondary_prob / primary_prob)

            #compare the total pixels. If it's a minor type it will be smaller
            if major_counts > minor_counts and major_type in self.best_surface_types:
                self.surfaceType = major_type
                self._alteration = "min %d->%d maj" % (minor_counts, major_counts)
            elif self.surfaceType.is_pair(self.secondaryType) and sp_ratio > 0.8:
                self._alteration = "expanded %.2f" % (sp_ratio)
                self.surfaceType = major_type

        if self._alteration is not None:
            print("Changed %d from %s to %s: %s" % (self.index, self.best_surface_types[0].name, self.surfaceType.name, self._alteration))

    def analyze(self, confidence=0.05, K=3):

        highest = np.max(self.probs)
        self.confidence = max(min(highest * 0.9, confidence), 0.05)

        self.determine_surface_type(K)


    def debug(self, img, color):
        cv2.drawContours(img, self.contours, -1, color)

        if self.surfaceType.is_major and self.surfaceType != SurfaceType.Other:
            Line.draw_all(img, self.lines, color=color, thickness=2)
    
        if self.center is None: 
            print("No center found for %s" % self.name)
            return

        pos = self.center
        pos = put_text(img, self.name, pos, color, size=0.5, shadow=True, highlights=True)

        if self.cloned_from >= 0:
            pos = put_text(img, "cloned %d" % self.cloned_from, pos, (255, 0, 0), size=0.33, shadow=True)

        if self._alteration is not None:
            pos = put_text(img, self._alteration, pos, color, size=0.33, shadow=True)

        pos = put_text(img, "%.0f deg" % np.degrees(self.angle), pos, (255, 255, 255), size=0.33, shadow=True)

        #print("%s has %d lines" % (self.name, len(self.lines)))

        # if surface.surfaceType != surface.best_surface_types[0]:
        #     best_prob = surface.category_probs[surface.best_surface_types[0]]
        #     chosen_prob = surface.category_probs[surface.surfaceType]
        #     text = "%s %.2f to %s %.2f" % (surface.best_surface_types[0].name, best_prob, surface.surfaceType.name, chosen_prob)

        #     put_text(img, text, (position[0], position[1] + text_size[1]), (255, 255, 255), size=0.33, shadow=True) 
