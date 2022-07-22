import numpy as np

import cv2
import random
import time
import math
from scipy.spatial import distance
from operator import attrgetter

from cambrian.LineFunctions import LineFunctions

from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_image, im_logging_enabled, log_markers, get_segmentation_image, log_segmentation_image, log_mask
from pipeline.components.line import draw_lines
from .vanishingpointfinder import angle_with_vp
from pipeline.misc.utils import random_color, resize_array, adjust_mask
from pipeline.data.ade20k import ADE20K, on_floor, on_wall, on_ceiling, box_like, legged_objects
from pipeline.components.rotated_rect import RotatedRect
from pipeline.components.line import Line, extend_to_intersection, draw_lines, line_within_mask, line_on_image_edge, merge_lines
from pipeline.stages.vanishingpointfinder import get_inliers

class PipelineBarrierFinder(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.FindBarriers

    @property
    def required_keys(self) -> list:
        return ["room", "downscaled"]

    @property
    def output_keys(self) -> list:
        return []


    def run(self, data):

        self.data = data
        self.image = self.data["downscaled"]
        self.room = self.data["room"]

        diagonal = math.hypot(self.image.shape[0], self.image.shape[1])
        min_line_length = diagonal / 30

        shape = (self.image.shape[1], self.image.shape[0])
        probs = resize_array(self.data["planes"]["masks"], shape)

        wall = np.zeros(self.image .shape[:2], dtype=np.uint8)
        wall[self.data["isolated_labels"] == SurfaceType.Wall.index] = 1
        wall[self.data["isolated_labels"] == SurfaceType.OnWall.index] = 1
        for label in box_like:
            wall[self.data["semantic_labels"] == label.index] = 1

        wall_mask = adjust_mask(cv2.dilate, wall, size=5, scale=0.4)

        index_mask = np.dstack(tuple(probs))
        index_mask = np.int32(np.argmax(index_mask, -1))

        # index_mask = self.data["index_mask"]

        all_planes = np.unique(index_mask).astype(np.int32)

        new_lines = []

        for plane_index in all_planes:

            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            mask[index_mask == plane_index] = 1

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                shape = contour.shape
                contour = (contour.flatten() - 1).reshape(shape)

                epsilon = 5
                polygon = cv2.approxPolyDP(contour, epsilon, True)

                num_pts = len(polygon)

                for i in range(num_pts):
                    point_a = polygon[i][0]
                    point_b = polygon[(i+1) % num_pts][0]

                    length = distance.euclidean(point_a, point_b)

                    if length >= min_line_length and not line_on_image_edge(point_a, point_b, self.image.shape[1], self.image.shape[0], min_distance=diagonal/50):
                        new_lines.append(Line(point_a[0], point_a[1], point_b[0], point_b[1], group="plane_lines"))


        plane_lines = list(get_inliers(new_lines, self.data["vertical_vp"].model, np.radians(15)))
        #plane_lines = new_lines
        plane_lines = list(filter(lambda line: line_within_mask(line, wall_mask), plane_lines))

        plane_lines = merge_lines(plane_lines, search_width=max(diagonal/50, 3), search_length=1.5, angle_threshold=np.radians(20))


        ceiling = np.zeros(self.image .shape[:2], dtype=np.uint8)
        ceiling[self.data["isolated_labels"] == SurfaceType.Ceiling.index] = 1
        ceiling[self.data["isolated_labels"] == SurfaceType.OnCeiling.index] = 1

        floor = np.zeros(self.image .shape[:2], dtype=np.uint8)
        floor[self.data["isolated_labels"] == SurfaceType.Floor.index] = 1
        floor[self.data["isolated_labels"] == SurfaceType.OnFloor.index] = 1

        
        other = np.zeros(self.image .shape[:2], dtype=np.uint8)
        other[self.data["isolated_labels"] == SurfaceType.Other.index] = 1 

        other = adjust_mask(cv2.dilate, other, size=11, scale=0.4)

        wall = adjust_mask(cv2.dilate, wall, size=11, scale=0.4)

        floor_ceiling = adjust_mask(cv2.dilate, cv2.bitwise_or(floor, ceiling), size=11, scale=0.4)

        search_mask = cv2.bitwise_and(floor_ceiling, wall)

        log_mask(self.data, "barriers_mask", search_mask, self.data["downscaled"])

        def filter_horizontal_lines(lines):

            vertical_lines = list(get_inliers(lines, data["vertical_vp"].model, np.radians(8)))
            horizontal_lines = list(set(lines).difference(vertical_lines))

            horizontal_lines = list(filter(lambda line: line_within_mask(line, search_mask), horizontal_lines))

            return horizontal_lines


        horizontal_semantic_lines = filter_horizontal_lines(self.data["semantic_lines"]) 
        horizontal_vp_lines = filter_horizontal_lines(self.data["vp_lines"]) 

        horizontal_lines = horizontal_semantic_lines + horizontal_vp_lines
        horizontal_lines = merge_lines(horizontal_lines.copy(), search_width=max(diagonal/80, 3), search_length_offset=diagonal/40, angle_threshold=np.radians(5))

        all_lines, intersections = extend_to_intersection(horizontal_lines, search_length=2.0, modify=False)


        if im_logging_enabled(data):
            debug = get_segmentation_image(self.data["isolated_labels"], self.data["downscaled"], labelset=None)
            debug = cv2.addWeighted(debug, 0.3, self.data["downscaled"], 0.7, 0)

            draw_lines(debug, plane_lines, color=(0,255,0), thickness=3, lineType=cv2.LINE_AA)
            draw_lines(debug, all_lines, color=(255,0,50), thickness=3, lineType=cv2.LINE_AA)

            draw_lines(debug, self.data["vp_lines"], color=(50,50,50), thickness=1)

            for point in intersections:
                x = int(max(point[0], 0))
                y = int(max(point[1], 0))

                cv2.circle(debug, (x, y), 5, (255, 180, 0), cv2.FILLED, cv2.LINE_AA)

            log_image(self.data, "barriers", debug)

