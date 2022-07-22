import numpy as np
from scipy import ndimage
import cv2

from skimage.morphology import skeletonize, remove_small_objects
from skimage.segmentation import join_segmentations, watershed
from skimage.morphology import disk
from skimage.filters import rank

import cambrian.image_processing as ip

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.surface_type import SurfaceType
from pipeline.components.line import Line, draw_lines
from .planegeometry import Dimension
from pipeline.misc.utils import random_color
from pipeline.data.logging import log_segmentation_image, im_logging_enabled, log_image, LogLevel, log_markers, Timer


class SurfaceRefinement():
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.room = self.data["room"]
        self.image = self.data["image"]
        
    def refine(self, use_HED=False):

        markers = np.zeros(self.image.shape[:2], dtype=np.int32)
        watershed_image = cv2.resize(self.data["hed"], (self.image.shape[1], self.image.shape[0]))
        watershed_mask = np.ones(markers.shape, dtype=np.int32)

        color = 1
        scale = self.image.shape[0] / self.data["downscaled"].shape[0]
        thickness = int(scale * 3)

        for surface in self.room.surfaces:
            
            contours, _ = cv2.findContours(surface.hires_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

            markers[surface.hires_mask > 0] = color
            cv2.drawContours(markers, contours, -1, 0, thickness)

            color += 1

        barrier_lines = list(map(lambda line: line.extended(1.1), self.data["semantic_lines"]))

        draw_lines(watershed_mask, barrier_lines, color=0, scale=scale, lineType=cv2.LINE_4)

        log_markers(self.data, "room_markers", markers, primary=True, num_labels=len(self.room.surfaces))

        markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))
        markers[markers<0] = 0

        log_markers(self.data, "room_markers_final", markers, primary=True, num_labels=len(self.room.surfaces))

        color = 1
        for surface in self.room.surfaces:
            mask = np.zeros_like(surface.hires_mask)
            mask[markers == color] = 1

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel)

            surface.hires_mask = mask
            color += 1

        log_image(self.data, "room_final", self.room.get_debug_image(hires=True))            
        
        
class PipelineSurfaceRefinement(PipelineStep):

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Refine

    @property
    def required_keys(self) -> list:
        return ["room"]

    @property
    def output_keys(self) -> list:
        return ["segmentation", "masks"]

    def run(self, data):
        refiner = SurfaceRefinement(data)
        refiner.refine()

        
