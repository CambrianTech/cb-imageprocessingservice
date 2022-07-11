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
from pipeline.misc.utils import get_segmentation_image, random_color, scale_contour
from pipeline.data.logging import log_segmentation_image, im_logging_enabled, log_image, LogLevel, log_markers, Timer

class SurfaceRefinement():
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.room = self.data["room"]
        self.image = self.data["image"]
        
    def refine(self, use_HED=False):

        self.data["lighting"] = cv2.edgePreservingFilter(np.uint8(self.data["lighting"]), flags=1, sigma_s=10, sigma_r=1.0)
        log_image(self.data, 'lighting_smooth', self.data["lighting"])

        markers = np.zeros(self.image.shape[:2], dtype=np.int32)
        watershed_image = cv2.resize(self.data["hed"], (self.image.shape[1], self.image.shape[0]))
        watershed_mask = np.zeros(markers.shape, dtype=np.int32)

        color = 1
        scale = self.image.shape[0] / self.data["downscaled"].shape[0]
        thickness = 5 + int(max(scale * 3, 2))

        for surface in self.room.surfaces:
            markers[surface.hires_mask > 0] = color
            watershed_mask[surface.hires_mask > 0] = 1

            contours, hierarchy = cv2.findContours(surface.hires_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(markers, contours, -1, 0, thickness)

            cv2.drawContours(watershed_mask, contours, -1, 1, cv2.FILLED)
            cv2.drawContours(watershed_mask, contours, -1, 1, thickness)

            color += 1

        vp_lines = []

        for vp in self.room.vanishing_points:
            vp_lines.extend(vp.inliers)


        # draw_lines(src, vp_lines, color=(0,0,0), scale=src_scale)
        # draw_lines(src, self.data["lines"], color=(255,0,255), scale=src_scale)
        #draw_lines(markers, vp_lines, color=255, scale=src_scale, lineType=cv2.LINE_4)

        log_markers(self.data, "room_markers", markers, primary=True, num_labels=len(self.room.surfaces))

        markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))
        markers[markers<0] = 0

        log_markers(self.data, "room_markers_final", markers, primary=True, num_labels=len(self.room.surfaces))


        color = 1
        for surface in self.room.surfaces:
            mask = np.zeros_like(surface.hires_mask)
            mask[markers == color] = 1
            surface.hires_mask = mask
            color += 1

        if im_logging_enabled(self.data):
            log_image(self.data, "room_final", self.room.get_debug_image(hires=True))

        #log_image(self.data, "room_refined", self.room.get_debug_image(hires=True))
            
        
        
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

        
