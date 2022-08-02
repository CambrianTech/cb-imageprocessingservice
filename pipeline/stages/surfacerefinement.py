import numpy as np
from scipy import ndimage
import cv2
import random

from skimage.segmentation import watershed

import cambrian.image_processing as ip

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.surface_type import SurfaceType
from pipeline.components.line import Line, draw_lines
from .planegeometry import Dimension
from pipeline.misc.utils import random_color
from pipeline.data.logging import log_segmentation_image, im_logging_enabled, log_image, LogLevel, log_markers, log_mask


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

        index_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        alpha = np.ones(self.image.shape[:2], dtype=np.uint8)
        num_surfaces = len(self.room.surfaces)

        #there's an issue with the shaders causing too close indices to mix up and creating artifacts
        #so giving spaced out indices randomly is a temporary solution
        indices = random.sample(range(1, 255), num_surfaces)

        for i, surface in enumerate(self.room.surfaces):

            color = i + 1
            mask = np.zeros_like(surface.hires_mask)
            mask[markers == color] = 1
            mask[index_mask > 0] = 0

            surface.hires_mask = mask

            surface.index = indices[i]

            index_mask[surface.hires_mask > 0] = surface.index

            log_mask(self.data, "surface_%d" % surface.index, mask, background=self.image)


        #index_mask = cv2.dilate(index_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)

        index_mask = np.int32(watershed(watershed_image, index_mask))

        # index_mask = cv2.cvtColor(index_mask, cv2.COLOR_GRAY2RGB)
        # alpha = np.reshape(alpha, (*alpha.shape,1))
        # index_mask = np.concatenate([index_mask, alpha], axis=2)

        self.data["index_mask"] = index_mask

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

        
