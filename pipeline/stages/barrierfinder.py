import numpy as np

import cv2
import random
import time
import math
from scipy.spatial import distance
from operator import attrgetter
from skimage.segmentation import watershed
import itertools

from cambrian.LineFunctions import LineFunctions

from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_image, im_logging_enabled, log_markers, log_segmentation_image, log_mask
from pipeline.misc.utils import convert_color, list_flatten
from pipeline.components.line import draw_lines
from .vanishingpointfinder import angle_with_vp
from pipeline.misc.utils import random_color
from pipeline.data.ade20k import ADE20K
from pipeline.stages.extractsurfaces import on_floor, on_wall, on_ceiling, box_like, legged_objects
from pipeline.components.rotated_rect import RotatedRect
from pipeline.components.line import extend_lines


class LabelFreedom():

    def __init__(self, labels, freedom):
        self.labels = list_flatten(labels)
        self.freedom = freedom

    def index_of(self, label):
        try:
            return self.labels.index(label)
        except:
            return -1

class PipelineBarrierFinder(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.FindBarriers

    @property
    def required_keys(self) -> list:
        return ["room", "downscaled", "isolated", "lines"]

    @property
    def output_keys(self) -> list:
        return []

    def refine_semantics(self, label_freedoms:list, name="semantic"):

        output = self.data["semantic_probs"]
        labels = np.argmax(np.dstack(output), -1)

        image = self.data["downscaled"]
        markers = np.zeros(image.shape[:2], dtype=np.int32)

        watershed_image = cv2.resize(self.data["hed"], (image.shape[1], image.shape[0]))
        watershed_mask = np.zeros(markers.shape, dtype=np.int32)

        min_matches = 100
        num_labels = np.amax(labels) + 1

        semantic_key = {}

        for label in range(0, num_labels):

            label_mask = labels == label

            value = label + 1

            match = next(filter(lambda x: x.index_of(value) >= 0, label_freedoms), None)

            freedom = match.freedom if match is not None else None

            if freedom is not None and len(labels[label_mask]) > min_matches:

                index = match.index_of(value)
                semantic_label = match.labels[index]

                semantic_key[value] = semantic_label.name

                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[label_mask] = 1
                watershed_mask[label_mask] = 1

                mask_padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                contours, _ = cv2.findContours(mask_padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    shape = contour.shape
                    contour = (contour.flatten() - 1).reshape(shape)

                    sub_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(sub_mask, [contour], -1, 1, thickness=cv2.FILLED)
                    cv2.drawContours(markers, [contour], -1, value, thickness=cv2.FILLED)

                    thickness = int(np.sqrt(cv2.contourArea(contour)) / 15)
                    cv2.drawContours(markers, [contour], -1, 0, thickness=thickness)

                    # # #padded transform
                    mask_padded = cv2.copyMakeBorder(sub_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                    dist_transform = cv2.distanceTransform(mask_padded, cv2.DIST_L2, 5)
                    dist_transform = dist_transform[1:-1,1:-1]

                    markers[dist_transform > freedom * dist_transform.max()] = value


        def is_barrier_line(line, mask, num_points=7, num_matches=3):
            line_points = np.linspace(line.point_a, line.point_b, num_points)
            count = 0
            for point in line_points:
                if point[0] < 0 or point[0] >= mask.shape[1] or point[1] < 0 or point[1] >= mask.shape[0]: continue

                if mask[int(point[1]), int(point[0])] == 0:
                    count += 1

                if count > num_matches:
                    return True

            return False
        
        line_candidates = list(filter(lambda line: is_barrier_line(line, markers), self.data["vp_lines"]))
        line_candidates = extend_lines(line_candidates)

        draw_lines(watershed_mask, line_candidates, color=0, thickness=1, lineType=cv2.LINE_4)


        log_mask(self.data, "%s_mask" % name, watershed_mask)
        log_segmentation_image(self.data, "%s_markers" % name, markers, self.data["downscaled"], labelset=semantic_key)

        markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))
        markers[markers<0] = 0

        # for label in range(1, num_labels):
        #     mask = np.zeros(image.shape[:2], dtype=np.uint8)
        #     mask[markers == color] = 1

        log_segmentation_image(self.data, "%s_refined" % name, markers, self.data["downscaled"], labelset=semantic_key)

    def run(self, data):

        self.data = data
        self.image = self.data["downscaled"]
        self.room = self.data["room"]

        self.refine_semantics([ LabelFreedom([ADE20K.ceiling], 0.2), LabelFreedom([ADE20K.wall], 0.2), LabelFreedom(box_like, 0.03), LabelFreedom(on_wall, 0.1)], name="wall_ceiling")
        self.refine_semantics([ LabelFreedom([ADE20K.wall], 0.02), LabelFreedom(box_like, 0.03), LabelFreedom([ADE20K.floor], 0.05), LabelFreedom([ADE20K.stairs, ADE20K.stairway], 0.05), LabelFreedom(on_floor, 0.05), LabelFreedom(legged_objects, 0.05)], name="floor")

