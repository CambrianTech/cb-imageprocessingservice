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
from pipeline.misc.utils import get_segmentation_image, random_color, adjust_mask
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
        watershed_mask = np.ones(markers.shape, dtype=np.int32)

        color = 1
        scale = self.image.shape[0] / self.data["downscaled"].shape[0]
        thickness = 5 + int(scale * 5)

        for surface in self.room.surfaces:
            
            contours, hierarchy = cv2.findContours(surface.hires_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            cv2.drawContours(markers, contours, -1, color, cv2.FILLED)
            #cv2.drawContours(watershed_mask, contours, -1, 1, cv2.FILLED)

            cv2.drawContours(markers, contours, -1, 0, thickness)
            #cv2.drawContours(watershed_mask, contours, -1, 1, thickness)

            color += 1

        vp_lines = []

        for vp in self.room.vanishing_points:
            vp_lines.extend(vp.inliers)


        # draw_lines(src, vp_lines, color=(0,0,0), scale=scale)
        # draw_lines(src, self.data["lines"], color=(255,0,255), scale=scale)

        barrier_lines = list(map(lambda line: line.extended(1.1), self.room.barrier_lines))
        draw_lines(watershed_mask, barrier_lines, color=0, scale=scale, lineType=cv2.LINE_4)

        #watershed_mask = adjust_mask(cv2.dilate, watershed_mask, size=3)

        log_markers(self.data, "room_markers", markers, primary=True, num_labels=len(self.room.surfaces))

        markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))
        markers[markers<0] = 0

        # markers = np.int32(watershed(watershed_image, markers))        
        # markers[markers<0] = 0

        # watershed_mask = adjust_mask(cv2.dilate, watershed_mask, size=9, scale=0.33)
        # markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))
        # markers[markers<0] = 0

        log_markers(self.data, "room_markers_final", markers, primary=True, num_labels=len(self.room.surfaces))

        color = 1
        for surface in self.room.surfaces:
            mask = np.zeros_like(surface.hires_mask)
            mask[markers == color] = 1

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.dilate(mask, kernel)

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

        
