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
from pipeline.data.ade20k import ADE20K, on_floor, on_wall, on_ceiling, box_like, legged_objects, lights
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

        #obtain plane vertical lines, major barriers between original planes:
        wall = np.zeros(self.image .shape[:2], dtype=np.uint8)
        wall[self.data["isolated_labels"] == SurfaceType.OnWall.index] = 1

        on_wall = np.zeros(self.image .shape[:2], dtype=np.uint8)
        on_wall[self.data["isolated_labels"] == SurfaceType.OnWall.index] = 1
        on_wall_expanded = adjust_mask(cv2.dilate, on_wall, size=5)

        wall_like_mask = np.zeros(self.image .shape[:2], dtype=np.uint8)
        for label in box_like:
            wall_like_mask[self.data["semantic_labels"] == label.index] = 1

        wall_like_expanded = adjust_mask(cv2.dilate, wall_like_mask, size=5)

        wall[on_wall > 0] = 1
        wall[wall_like_mask > 0] = 1
        wall_expanded = adjust_mask(cv2.dilate, wall, size=5)
        wall_contracted = adjust_mask(cv2.erode, wall, size=5)

        index_mask = np.dstack(tuple(probs))
        index_mask = np.int32(np.argmax(index_mask, -1))

        all_planes = np.unique(index_mask).astype(np.int32)

        vertical_plane_lines = []

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
                        vertical_plane_lines.append(Line(point_a[0], point_a[1], point_b[0], point_b[1], group="plane_lines"))


        vertical_plane_lines = list(get_inliers(vertical_plane_lines, self.data["vertical_vp"].model, np.radians(15)))
        #vertical_plane_lines = list(filter(lambda line: line_within_mask(line, wall_expanded), vertical_plane_lines))
        vertical_plane_lines = merge_lines(vertical_plane_lines, search_width=max(diagonal/50, 3), search_length=0.9, angle_threshold=np.radians(20))

        ceiling = np.zeros(self.image .shape[:2], dtype=np.uint8)
        ceiling[self.data["isolated_labels"] == SurfaceType.Ceiling.index] = 1
        ceiling[self.data["isolated_labels"] == SurfaceType.OnCeiling.index] = 1
        ceiling_expanded = adjust_mask(cv2.dilate, ceiling, size=5, scale=0.5)

        on_ceiling = np.zeros(self.image .shape[:2], dtype=np.uint8)
        on_ceiling[self.data["isolated_labels"] == SurfaceType.OnCeiling.index] = 1
        on_ceiling_objects = [ADE20K.chandelier, ADE20K.fan] + lights
        for label in on_ceiling_objects:
            on_ceiling[self.data["semantic_labels"] == label.index] = 1
        on_ceiling = adjust_mask(cv2.dilate, on_ceiling, size=5, scale=0.5)

        floor = np.zeros(self.image .shape[:2], dtype=np.uint8)
        floor[self.data["isolated_labels"] == SurfaceType.Floor.index] = 1
        floor[self.data["isolated_labels"] == SurfaceType.OnFloor.index] = 1


        def partition_lines(lines, mask=None):

            vertical_lines = list(get_inliers(lines, self.data["vertical_vp"].model, np.radians(8)))
            horizontal_lines = list(set(lines).difference(vertical_lines))

            # if mask is not None:
            #     horizontal_lines = list(filter(lambda line: line_within_mask(line, mask), horizontal_lines))

            return horizontal_lines, vertical_lines

        #grab all horizontal lines from semantic lines, which is approximately the edges of surfaces, but not between walls.
        #then also combine these with nearby horizontal lines
        horizontal_semantic_lines, vertical_semantic_lines = partition_lines(self.data["semantic_lines"])
        horizontal_semantic_lines = list(filter(lambda line: line_within_mask(line, ceiling_expanded) \
                                    and not line_within_mask(line, on_ceiling) and not line_within_mask(line, wall_contracted), horizontal_semantic_lines))

        horizontal_semantic_lines = merge_lines(horizontal_semantic_lines, search_width=max(diagonal/100, 3), search_length=1.3, angle_threshold=np.radians(5))

        horizontal_vp_lines, vertical_vp_lines = partition_lines(self.data["vp_lines"])

        #all meaningful horizontal lines
        horizontal_lines = horizontal_semantic_lines + horizontal_vp_lines
        vertical_lines = vertical_semantic_lines + vertical_vp_lines

        #re-cluster the original horizontal lines and plane vertical lines + vertical lines (remove_matches=False):
        def cluster_matches(src_lines, lines, search_width, search_length=1.3, angle_threshold=np.radians(5)):

            for line in src_lines + lines:
                line.cluster = None

            cluster_index = 0

            for src_line in src_lines:
                src_line.cluster = cluster_index

                src_rect = src_line.bounding_box(width=search_width, length_multiplier=search_length)

                for line in lines:

                    if line_within_mask(line, wall_like_expanded) or line_within_mask(line, on_wall_expanded):
                        continue

                    #maybe use vanishing points instead?
                    if LineFunctions.line_angle_difference(src_line.angle, line.angle) > angle_threshold:
                        continue

                    rect = line.bounding_box(width=3)

                    result, _ = cv2.rotatedRectangleIntersection(src_rect, rect)

                    if result != 0:
                        line.cluster = cluster_index

                cluster_index += 1

        cluster_matches(horizontal_semantic_lines, horizontal_vp_lines, diagonal/15)

        #find intersections of grouped lines, being careful not to corrupt originals (copy)
        #horizontal_lines_merged = merge_lines(horizontal_lines.copy(), search_width=max(diagonal/150, 3), angle_threshold=np.radians(5), modify=False)

        #extend these merged horizontal lines together to find intersections 
        horizontal_guide_lines, intersections = extend_to_intersection(horizontal_semantic_lines, search_length=2.0, modify=False)

        horizontal_clusters = list(set([line.cluster for line in horizontal_semantic_lines]))
        adjacent_horizontal_lines = list(filter(lambda line: line.cluster in horizontal_clusters, horizontal_lines))

        vertical_clusters = list(set([line.cluster for line in vertical_plane_lines]))
        adjacent_vertical_lines = list(filter(lambda line: line.cluster in vertical_clusters, vertical_lines))        

        if im_logging_enabled(data):
            #debug = get_segmentation_image(self.data["isolated_labels"], self.data["downscaled"], labelset=None)\
            debug = self.data["downscaled"].copy()

            draw_lines(debug, vertical_plane_lines, color=(0,255,0), thickness=5)
            draw_lines(debug, horizontal_guide_lines, color=(255,0,50), thickness=3)

            debug = cv2.addWeighted(debug, 0.7, self.data["downscaled"], 0.3, 0)

            draw_lines(debug, self.data["vp_lines"], color=(50, 50, 50), thickness=1)

            draw_lines(debug, adjacent_vertical_lines, color=(0, 255, 0), thickness=1)
            draw_lines(debug, adjacent_horizontal_lines, color=(255, 0, 0), thickness=1)

            # draw_lines(debug, self.data["vp_lines"], color=(50,50,50), thickness=1)

            for point in intersections:
                x = int(max(point[0], 0))
                y = int(max(point[1], 0))

                cv2.circle(debug, (x, y), 5, (255, 180, 0), cv2.FILLED, cv2.LINE_AA)

            log_image(self.data, "barriers", debug)

