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
from pipeline.data.logging import log_image, im_logging_enabled, log_markers, get_segmentation_image, log_segmentation_image
from pipeline.components.line import draw_lines
from .vanishingpointfinder import angle_with_vp
from pipeline.misc.utils import random_color, resize_array
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

                    if length >= min_line_length and not line_on_image_edge(point_a, point_b, self.image.shape[1], self.image.shape[0], min_distance=3):
                        self.room.vertical_vp.model

                        new_lines.append(Line(point_a[0], point_a[1], point_b[0], point_b[1], group="plane_lines"))


        vertical_lines = list(get_inliers(new_lines, self.data["vertical_vp"].model, np.radians(8)))

        vertical_lines = merge_lines(vertical_lines, search_width=max(diagonal/50, 3), search_length=1.0, angle_threshold=np.radians(20))

        # horizontal_lines = []
        # for vp in self.data["horizontal_vps"]:
        #     horizontal_lines.extend(list(get_inliers(new_lines, vp.model, np.radians(15))))

        # horizontal_lines = merge_lines(horizontal_lines, search_width=max(diagonal/50, 3), search_length=0.7, angle_threshold=np.radians(20))
        

        planes_debug = get_segmentation_image(index_mask, self.data["downscaled"])
        planes_debug = cv2.addWeighted(planes_debug, 0.5, self.data["downscaled"], 0.5, 0)
        draw_lines(planes_debug, vertical_lines, color=(255,0,0), thickness=2, lineType=cv2.LINE_AA)
        #draw_lines(planes_debug, horizontal_lines, color=(255,255,0), thickness=2, lineType=cv2.LINE_AA)

        log_image(self.data, "barriers_plane_lines", planes_debug)

        all_lines = self.data["vp_lines"]

        for surfaceType in [SurfaceType.Wall]:
            mask = np.zeros(self.image .shape[:2], dtype=np.uint8)
            mask[self.data["isolated_labels"] == surfaceType] = 1


        all_lines, intersections = extend_to_intersection(all_lines, search_length=1.1) 

        if im_logging_enabled(data):
            debug = get_segmentation_image(self.room.index_mask, self.data["downscaled"], labelset=None)
            debug = cv2.addWeighted(debug, 0.5, self.data["downscaled"], 0.5, 0)
            draw_lines(debug, all_lines, color=(255,0,50), thickness=2)
            for point in intersections:
                cv2.circle(debug, (int(point[0]), int(point[1])), 3, (255, 255, 0), cv2.FILLED, cv2.LINE_AA)

            log_image(self.data, "barriers", debug)