import numpy as np
from scipy import ndimage
import cv2
import random
import time
from enum import IntEnum
from scipy.spatial import distance
import math

from pipeline.data.surface_type import SurfaceType
from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.misc.utils import random_color, overlay_mask, convert_color
from .extractsurfaces import box_like, legged_objects
from .vanishingpointfinder import angle_with_vp
from pipeline.data.logging import log_image, log_segmentation_image, im_logging_enabled, log_markers
from cambrian.LineFunctions import LineFunctions
from pipeline.components.line import line_angle_difference, Line, line_on_image_edge, merge_lines, get_line_points, draw_line
from pipeline.components.rotated_rect import RotatedRect
from pipeline.components.surface import Surface, surface_surface_key

class Barrier():
    def __init__(self, uniqueId, surface_a, surface_b):

        self.uniqueId = uniqueId
        self.surface_a = surface_a
        self.surface_b = surface_b

    def solve(self, img):
        self.contours = self.surface_a.intersection(self.surface_b)

        diagonal = int(math.hypot(img.shape[0], img.shape[1]))

        self.mask = np.zeros(img.shape[:2], dtype=np.uint8)
        cv2.drawContours(self.mask, self.contours, -1, 255, cv2.FILLED)
        self.skeleton = cv2.ximgproc.thinning(self.mask, cv2.ximgproc.THINNING_GUOHALL) #way better than skimage.morphology.skeletonize
        self.skeleton_contours, hierarchy = cv2.findContours(self.skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        epsilon = diagonal / 200
        self.skeleton_contours = list(map(lambda contour: cv2.approxPolyDP(contour, epsilon, True), self.skeleton_contours))



    def debug(self, img, hue=20):

        color = convert_color((hue, 255, 255), cv2.COLOR_HSV2RGB_FULL)

        cv2.drawContours(img, self.contours, -1, color, 1)

        cv2.drawContours(img, self.skeleton_contours, -1, color, 3)

        return img
        
        
class PipelineBarrierFinder(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Barriers

    @property
    def required_keys(self) -> list:
        return ["room"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):
        self.data = data
        self.image = data["downscaled"]
        self.room = data["room"]

        self.surfaces = []
        self.surfaces.extend(self.room.get_surfaces(surfaceTypes=[SurfaceType.Wall, SurfaceType.Floor, SurfaceType.Ceiling]))
        self.surfaces.extend(self.room.get_surfaces(labels=box_like))

        self.barriers = {}

        #get all barriers
        for surface in self.surfaces:
            for neighbor in surface.neighbors:

                if neighbor not in self.surfaces and not neighbor.surfaceType.is_pair(surface.surfaceType): continue
                #if not (neighbor in self.surfaces or neighbor.surfaceType.is_pair(surface.surfaceType)): continue

                uniqueId = surface_surface_key(surface, neighbor)

                if uniqueId not in self.barriers:
                    self.barriers[uniqueId] = Barrier(uniqueId, surface, neighbor)

        #solve
        for uniqueId in self.barriers:
            self.barriers[uniqueId].solve(self.image)


        log_image(self.data, "barriers.png", self.get_debug_image())

    def get_debug_image(self):

        img = self.image.copy()

        #draw masks
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)

        hues = random.sample(range(0, 360), len(self.surfaces))

        index = 0
        for surface in self.surfaces:
            pixels = surface.mask > 0
            img_hsv[:, :, 0][pixels] = hues[index]
            img_hsv[:, :, 1][pixels] = 127
            index += 1

        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)

        hues = random.sample(range(0, 360), len(self.barriers))

        index = 0
        for uniqueId in self.barriers:
            img = self.barriers[uniqueId].debug(img, hues[index])
            index += 1


        

        return img

def closest_polygon_side(contour, point, min_length_threshold=None, max_length_threshold=None):
    #return (distance, point, and indices) of closest line in polygon or contour by midpoints

    num_pts = len(contour)
    min_dist_sq = np.inf
    min_index = None

    min_length_threshold_sq = None if min_length_threshold is None else min_length_threshold * min_length_threshold
    max_length_threshold_sq = None if max_length_threshold is None else max_length_threshold * max_length_threshold

    for i in range(num_pts):
        point_a = contour[i][0]
        point_b = contour[(i+1) % num_pts][0]

        if point_a[0] == point_b[0] and point_a[1] == point_b[1]: continue

        if min_length_threshold_sq is not None or max_length_threshold_sq is not None:
            length_sq = distance.sqeuclidean(point_a, point_b)

            if min_length_threshold_sq is not None and length_sq < min_length_threshold_sq: continue
            if max_length_threshold_sq is not None and length_sq > max_length_threshold_sq: continue

        midpoint = (point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2

        #midpoints between point a and midpoint
        point_a = (point_a[0] + midpoint[0]) / 2, (point_a[1] + midpoint[1]) / 2
        point_b = (point_b[0] + midpoint[0]) / 2, (point_b[1] + midpoint[1]) / 2

        dist_point_a_sq = distance.sqeuclidean(point_a, point)
        dist_point_b_sq = distance.sqeuclidean(point_b, point)
        dist_midpoint_sq = distance.sqeuclidean(midpoint, point)

        if dist_point_a_sq < min_dist_sq:
            min_index = i
            min_dist_sq = dist_point_a_sq

        if dist_point_b_sq < min_dist_sq:
            min_index = i
            min_dist_sq = dist_point_b_sq

        if dist_midpoint_sq < min_dist_sq:
            min_index = i
            min_dist_sq = dist_midpoint_sq

    if min_index is None:
        if min_length_threshold is not None:
            return closest_polygon_side(contour, point, min_length_threshold=None, max_length_threshold=max_length_threshold)
        elif max_length_threshold is not None:
            return closest_polygon_side(contour, point, min_length_threshold=None, max_length_threshold=None)
        
    return min_index, np.sqrt(min_dist_sq)
