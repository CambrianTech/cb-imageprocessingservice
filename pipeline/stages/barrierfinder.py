import numpy as np

import cv2
import random
import time
import math
from scipy.spatial import distance
from operator import attrgetter
from skimage.segmentation import watershed

from cambrian.LineFunctions import LineFunctions
from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_image, im_logging_enabled, log_markers, log_segmentation_image
from pipeline.misc.utils import convert_color
from pipeline.components.line import draw_lines
from .vanishingpointfinder import angle_with_vp
from pipeline.misc.utils import random_color
from pipeline.data.ade20k import ADE20K
from pipeline.stages.extractsurfaces import on_floor, on_wall, on_ceiling, box_like, legged_objects
from pipeline.components.rotated_rect import RotatedRect

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

    def refine_semantics(self, label_freedoms:list):

        output = self.data["semantic_probs"]
        labels = np.argmax(np.dstack(output), -1)

        image = self.data["downscaled"]
        markers = np.zeros(image.shape[:2], dtype=np.int32)
        watershed_image = cv2.resize(self.data["hed"], (image.shape[1], image.shape[0]))
        watershed_mask = np.ones(markers.shape, dtype=np.int32)

        min_matches = 100
        num_labels = np.amax(labels) + 1
        freedom = 0.07

        color = 1
        for label in range(0, num_labels):

            label_mask = labels == label

            value = label + 1

            match = next(filter(lambda x: value in x[0], label_freedoms), None)

            freedom = match[1] if match is not None else None

            if freedom is not None and len(labels[label_mask]) > min_matches:
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[label_mask] = 1

                #padded transform
                mask_padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                dist_transform = cv2.distanceTransform(mask_padded, cv2.DIST_L2, 5)
                dist_transform = dist_transform[1:-1,1:-1]

                # if value in [ADE20K.ceiling, ADE20K.wall]:
                #     freedom = 0.2
                # elif value in on_wall:
                #     freedom = 0.05
                # elif value in [ADE20K.floor]:
                #     freedom = 0.03
                # elif value in on_floor:
                #     freedom = 0.03
                # elif value in box_like:
                #     freedom = 0.03
                # elif value in on_ceiling:
                #     freedom = 0.05
                # else:
                #     watershed_mask[label_mask] = 0
                #    continue

                markers[dist_transform > freedom * dist_transform.max()] = color
                color += 1

        log_markers(self.data, "semantic_refinement_markers", markers, primary=True, num_labels=color)

        markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))
        markers[markers<0] = 0

        log_segmentation_image(self.data, "semantic_refinement", markers, self.data["downscaled"])

    def run(self, data):

        self.data = data
        self.image = self.data["downscaled"]
        self.room = self.data["room"]

        self.refine_semantics([ ([ADE20K.ceiling, ADE20K.wall], 0.2), ([ADE20K.floor, box_like], 0.03) ])

