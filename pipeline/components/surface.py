import abc
from abc import ABCMeta, abstractmethod
import numpy as np
import math
from scipy import ndimage
import cv2
import uuid
from copy import copy, deepcopy
from termcolor import colored

import random

from pipeline.data.surface_type import SurfaceType
from pipeline.misc.utils import convert_color, put_text, sample_at_point
from .line import Line
from pipeline.data.ade20k import ADE20K
from pipeline.components.rotated_rect import RotatedRect
from pipeline.misc.utils import adjust_mask, scale_contour

from pipeline.stages.vanishingpointfinder import get_inliers

mask_padding = 10

class SurfaceBarrier():
    def __init__(self, vp, lines):
        self.vp = vp
        self.lines = lines

class Surface():

    def __init__(self, data, index=None, surfaceType=None):
        self.data = data
        self._index = index
        self.uniqueId = uuid.uuid4()
        self.surfaceType = surfaceType
        self.geometry = None
        self._mask = None

        self._mask_edges = None
        self._mask_expanded = None
        self._surface_mask = None
        self._mask_transform = None
        self._outer_mask = None
        self._inner_mask = None
        self._hires_mask = None

        self._alteration = None
        self.destroyed = False
        self._cloned_from = -1
        self._plane_mask = None
        self._contours = None
        self.moments = None
        self._polygons = None
        self._normals_color = None
        self._lines = None
        self._neighbors = None
        self.parent = None
        self.barriers = None

        self._semantic_labels = None

        self._plane_data = None
        self._plane_parameters = None
        self._normal = None
        self._offset = None
        self._axisRotation = None

        self._rotation = None

        self._background_mean_stddev = None
        self._lighting_mean_stddev = None

        self._normals_accumulated = None
        self._normals_mean = None
        self._mask_coordinates = None

        self._clusters = []
        self._hvps = None

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

    @property
    def probs(self) -> ndimage:
        probs_number = len(self.geometry.probs)
        return self.geometry.probs[self.index] if self.index in range(probs_number) else self.geometry.probs[self._cloned_from]

    @property
    def plane_mask(self) -> ndimage:
        if self._plane_mask is None:
            self._plane_mask = np.zeros_like(self.geometry.index_mask)
            self._plane_mask[self.geometry.index_mask == self.index] = 1
        return self._plane_mask

    @property
    def plane_data(self) -> np.ndarray:
        if self._plane_data is None and self.index in range(len(self.data["planes"]["detection"])):
            self._plane_data = self.data["planes"]["detection"][self.index]
        return self._plane_data

    @property
    def plane_parameters(self) -> tuple:
        if self._plane_parameters is None:
            self._plane_parameters = np.array(self.plane_data[6:9], dtype=np.float32)
        return self._plane_parameters

    @property
    def normal(self) -> np.ndarray:
        if self._normal is None:
            self._normal = self.plane_parameters / np.maximum(self.offset, 1e-4)
        return self._normal

    @normal.setter
    def normal(self, value:np.ndarray):
        self._normal = value

    @property
    def offset(self) -> float:
        if self._offset is None:
            self._offset = np.linalg.norm(self.plane_parameters, axis=-1, keepdims=True)
        return self._offset

    @offset.setter
    def offset(self, value:float):
        self._offset = value

    @property
    def axisRotation(self) -> float:
        if self._axisRotation is None:
            #todo: this should be set directly instead
            if self.surfaceType in [SurfaceType.Floor, SurfaceType.OnFloor]:
                self._axisRotation = self.data["floor_rotation"]
            elif self.surfaceType in [SurfaceType.Ceiling, SurfaceType.OnCeiling]:
                self._axisRotation = -1 * self.data["floor_rotation"]
            else:
                self._axisRotation = 0

        return self._axisRotation

    @axisRotation.setter
    def axisRotation(self, value:float):
        self._axisRotation = value

    @property
    def normals_color(self) -> tuple:
        if self._normals_color is None and len(self.mask) > 0:

            mask_sample = sample_at_point(self.mask, point=self.center, size=100)
            sample = sample_at_point(self.geometry.normals, point=self.center, size=100)

            if cv2.countNonZero(mask_sample) > 10:
                self._normals_color = cv2.mean(sample, mask_sample)[:3]
            else:
                self._normals_color = cv2.mean(self.geometry.normals, self.mask)[:3]

        return self._normals_color

    @property
    def background_mean_stddev(self) -> tuple:
        if self._background_mean_stddev is None:
            self._background_mean_stddev = cv2.meanStdDev(self.data["downscaled"], self.mask)

        return self._background_mean_stddev

    @property
    def background_mean(self) -> tuple:
        return tuple(self.background_mean_stddev[0].flatten())

    @property
    def background_stddev(self) -> tuple:
        return tuple(self.background_mean_stddev[1].flatten())

    @property
    def lighting_mean_stddev(self) -> tuple:
        if self._lighting_mean_stddev is None:
            self._lighting_mean_stddev = cv2.meanStdDev(self.data["lighting"], self.mask)

        return self._lighting_mean_stddev

    @property
    def lighting_mean(self) -> tuple:
        return tuple(self.lighting_mean_stddev[0].flatten())

    @property
    def lighting_stddev(self) -> tuple:
        return tuple(self.lighting_mean_stddev[1].flatten())

    @property
    def lines(self) -> list:

        def is_inside(point):
            value = self.mask_expanded[int(point[1]), int(point[0])]
            return value > 0

        if self._lines is None:
            self._lines = []
            self._clusters = []

            for line in self.data["lines"]:

                if line.cluster in self._clusters:
                    continue

                midpoint_a = ((line.point_a[0] + line.midpoint[0]) / 2, (line.point_a[1] + line.midpoint[1]) / 2)
                midpoint_b = ((line.point_b[0] + line.midpoint[0]) / 2, (line.point_b[1] + line.midpoint[1]) / 2)                

                if is_inside(line.midpoint) and (is_inside(midpoint_a) or is_inside(midpoint_b)):
                    self._clusters.append(line.cluster)

            self._lines  = [line for line in self.data["lines"] if line.cluster in self._clusters]
        return self._lines

    @property
    def line_clusters(self):
        self.lines
        return self._clusters

    @property
    def width(self):
        self.contours
        return self._width

    @property
    def height(self):
        self.contours
        return self._height

    @property
    def max_area(self) -> list:
        self.contours
        return self._max_area

    @property
    def horizontal_vps(self) -> list:

        if self._hvps is None:
            self._hvps = []

            num_matches = []
            if len(self.lines) > 1:
                for vp in self.data["room"].horizontal_vps:
                    matches = get_inliers(self.lines, vp.model, angle_threshold=np.radians(5))
                    num_matches.append(len(matches))

            if len(num_matches) > 0:
                threshold = max(3, 2 * max(num_matches) / 3)
                
                for i in range(0, len(self.data["room"].horizontal_vps)):
                    if num_matches[i] > threshold:
                        self._hvps.append(self.data["room"].horizontal_vps[i])

        return self._hvps

    @property
    def neighbors(self) -> list:

        if self._neighbors is None:
            self._neighbors = []

            #probably many ways this can be optimized: downsized mask, countNonZero, etc.

            for candidate in self.geometry.surfaces:
                if candidate == self: continue
        
                #check for self in candidate to save time, or check for overlap
                if (candidate._neighbors is not None and self in candidate._neighbors):
                    self._neighbors.append(candidate)
                else:
                    intersection = self.geometry.surface_surface_intersection(self, candidate)
                  
                    if cv2.countNonZero(intersection) > 10: 
                        # print("intersection:", cv2.countNonZero(intersection))
                        self._neighbors.append(candidate)

        return self._neighbors

    #line intersection with other surface/plane, if any
    def intersection(self, surface) -> RotatedRect:
        if surface not in self.neighbors:
            return None

        masks_intersection = self.geometry.surface_surface_intersection(self, surface)
  
        contours, hierarchy = cv2.findContours(masks_intersection, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) == 0:
            return None

        return contours

        # rect = RotatedRect(cv2.minAreaRect(cnt))

        # if not rect.empty: 
        #     return rect

        # return None

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

    @property
    def surface_mask(self):
        if self._surface_mask is None:
            self._surface_mask = cv2.copyMakeBorder(self.mask, mask_padding, mask_padding, mask_padding, mask_padding, cv2.BORDER_CONSTANT, value=0)
        return self._surface_mask

    @property
    def mask_transform(self):
        if self._mask_transform is None:
            trans = cv2.distanceTransform(self.surface_mask, cv2.DIST_L2, 5)
            self._mask_transform = trans[mask_padding:-mask_padding,mask_padding:-mask_padding]

        return self._mask_transform

    @property
    def mask_expanded(self):
        if self._mask_expanded is None:
            self._mask_expanded = cv2.bitwise_or(self.mask, self.mask_edges)
        return self._mask_expanded

    @property
    def mask_edges(self):
        self.contours
        return self._mask_edges

    @property
    def hires_mask(self) -> ndimage:
        if self._hires_mask is None:
            self._hires_mask = cv2.resize(self.mask, (self.data["image"].shape[1], self.data["image"].shape[0]), interpolation=cv2.INTER_NEAREST)
        return self._hires_mask

    def set_hires_mask(self, mask):
        self._hires_mask = mask

    def mask_changed(self):
        self._lines = None
        self._contours = None
        self._polygons = None
        self._semantic_labels = None
        self._normals_color = None
        self._neighbors = None
        self._mask_edges = None
        self._mask_expanded = None
        self._normals_accumulated = None
        self._normals_mean = None

        self._surface_mask = None
        self._mask_transform = None
        self._outer_mask = None
        self._inner_mask = None
        self._hires_mask = None
        
        self._hvps = None

    @property
    def contours(self):
        if self._contours is None or self.moments is None:

            self._max_area = 0
            self._width = 0
            self._height = 0

            #make a 1 pixel border so that edge contours aren't zero area
            mask_bordered = cv2.copyMakeBorder(self.mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0) 
            _contours, self.hierarchy = cv2.findContours(mask_bordered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            self._mask_edges = np.zeros(self.mask.shape, dtype="uint8")

            #remove border offset:
            self._contours = []
            for contour in _contours:
                shape = contour.shape
                contour = (contour.flatten() - 1).reshape(shape)

                area = cv2.contourArea(contour)
                self._max_area = max(area, self._max_area)

                x,y,w,h = cv2.boundingRect(contour)
                self._width = max(self._width, w)
                self._height = max(self._height, h)

                padding = max(area / 10, 5)

                cv2.drawContours(self._mask_edges, [contour], -1, 1, thickness=20)

                self._contours.append(contour)

            self.moments = cv2.moments(self.mask)
            
            # self.moments = cv2.moments(self._contours[0]) if len(self._contours) > 0 else None

            # if self.moments is None or self.moments["m00"] == 0:
            #     self.moments = cv2.moments(self.mask)


        return self._contours

    @property
    def polygons(self):
        if self._polygons is None:
            # epsilon = math.hypot(self.data["downscaled"].shape[0], self.data["downscaled"].shape[1]) / 400
            # epsilon = 0.001*cv2.arcLength(contour,True)
            self._polygons = list(map(lambda contour: cv2.approxPolyDP(contour, 1, True), self.contours))

        return self._polygons

    @property
    def semantic_labels(self) -> ndimage:
        if self._semantic_labels is None:
            # self._semantic_labels = self.get_semantic_labels(self.mask)
            segments, counts = np.unique(self.geometry.semantic_labels[self.mask > 0], return_counts=True)
            segmentList = zip(segments.tolist(), counts.tolist())
            self._semantic_labels = sorted(segmentList, key=lambda x:x[1], reverse=True)

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
        self.destroyed = True
        #Important! do not add code here, add inside remove_surface, and call public methods on this object

    @property
    def center(self) -> tuple:
        #todo: use 2D projection
        if self.moments is None or self.moments["m00"] == 0:
            #print("No moments")
            cX = self.mask.shape[1] // 2
            cY = self.mask.shape[0] // 2
        else:
            cX = int(self.moments["m10"] / self.moments["m00"])
            cY = int(self.moments["m01"] / self.moments["m00"])
        return cX, cY

    @property
    def normals_accumulated(self) -> tuple:
        if self._normals_accumulated is None and len(self.mask) > 0:
            self._normals_accumulated = np.sum(self.data["room"].normals[self.mask > 0], axis=0)

        return self._normals_accumulated

    @property
    def normals_mean(self) -> tuple:
        if self._normals_mean is None and len(self.mask) > 0:
            self._normals_mean = self.normals_accumulated / cv2.countNonZero(self.mask)

        return self._normals_mean

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
            if angle_with_wall < angle_threshold and self.surfaceType != SurfaceType.Wall and self.surfaceType != SurfaceType.OnWall:
                self.surfaceType = SurfaceType.Wall if self.surfaceType.is_major else SurfaceType.OnWall
                self._alteration = "%d deg from wall" % int(math.degrees(angle_with_wall))
            elif angle_with_ceiling < angle_threshold and self.surfaceType != SurfaceType.Ceiling and self.surfaceType != SurfaceType.OnCeiling:
                self.surfaceType = SurfaceType.Ceiling if self.surfaceType.is_major else SurfaceType.OnCeiling
                self._alteration = "%d deg from ceil" % int(math.degrees(angle_with_ceiling))
            elif angle_with_floor < angle_threshold and self.surfaceType != SurfaceType.Floor and self.surfaceType != SurfaceType.OnFloor:
                self.surfaceType = SurfaceType.Floor if self.surfaceType.is_major else SurfaceType.OnFloor
                self._alteration = "%d deg from floor" % int(math.degrees(angle_with_floor))

            if was_ceiling and self.surfaceType != SurfaceType.Ceiling:
                self.destroy() #too problematic

        #If it is minor type, e.g. on wall or on floor, it may need to become a major type such as wall or floor:
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

    def analyze(self, confidence=0.04, K=3):

        highest = np.max(self.probs)
        self.confidence = max(min(highest * 0.9, confidence), 0.04)

        self.determine_surface_type(K)


    def debug(self, img, color, draw_contours=False):

        scale = img.shape[0] / self.data["downscaled"].shape[0]

        if draw_contours:
            if scale != 1.0:
                contours = []
                for contour in self.contours:
                    contours.append((contour * scale).astype(np.int32))
            else:
                contours = self.contours

            cv2.drawContours(img, contours, -1, color)

        # if self.surfaceType.is_major and self.surfaceType != SurfaceType.Other:
        #     Line.draw_all(img, self.lines, color=color)
        
        if self.center is None: 
            print("No center found for %s" % self.name)
            return
                
        if self.surfaceType in [SurfaceType.Wall, SurfaceType.Floor, SurfaceType.Ceiling]:
            pos = self.center[0] * scale, self.center[1] * scale
            pos = put_text(img, self.name, pos, color, size=0.5 * scale, shadow=True, highlights=True)

        # if self.cloned_from >= 0:
        #     pos = put_text(img, "cloned %d" % self.cloned_from, pos, (255, 0, 0), size=0.33 * scale, shadow=True)

        # if self._alteration is not None:
        #     pos = put_text(img, self._alteration, pos, color, size=0.33 * scale, shadow=True)

        # if self._plane_data is not None:
        #     pos = put_text(img, "%.0f deg" % np.degrees(self.angle), pos, (255, 255, 255), size=0.33 * scale, shadow=True)

        #print("%s has %d lines" % (self.name, len(self.lines)))

        # if surface.surfaceType != surface.best_surface_types[0]:
        #     best_prob = surface.category_probs[surface.best_surface_types[0]]
        #     chosen_prob = surface.category_probs[surface.surfaceType]
        #     text = "%s %.2f to %s %.2f" % (surface.best_surface_types[0].name, best_prob, surface.surfaceType.name, chosen_prob)

        #     put_text(img, text, (position[0], position[1] + text_size[1]), (255, 255, 255), size=0.33, shadow=True) 


