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
from pipeline.misc.utils import random_color
from pipeline.data.ade20k import ADE20K, on_floor, on_wall, on_ceiling, box_like, legged_objects
from pipeline.components.rotated_rect import RotatedRect
from pipeline.components.line import extend_to_intersection, draw_lines, line_within_mask

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