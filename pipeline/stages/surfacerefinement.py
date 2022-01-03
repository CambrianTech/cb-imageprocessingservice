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
from pipeline.components.line import Line
from .planegeometry import Dimension
from pipeline.misc.utils import get_segmentation_image, random_color
from pipeline.data.logging import log_segmentation_image, im_logging_enabled, log_image, LogLevel, log_markers

class SurfaceRefinement():
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.room = self.data["room"]
        self.image = self.data["image"]
        self.barriers = self.data["barriers"]
        
    def refine(self):
        self.data["lighting"] = cv2.edgePreservingFilter(np.uint8(self.data["lighting"]), flags=1, sigma_s=10, sigma_r=1.0)
        log_image(self.data, 'lighting_smooth', self.data["lighting"])

        masks = [cv2.resize(s.mask, (self.image.shape[1], self.image.shape[0])) for s in self.room.surfaces]
        total_mask = np.sum(np.dstack(masks), axis=-1, dtype=np.uint8)
        disputed_areas = np.zeros(total_mask.shape, dtype=np.uint8)
        disputed_areas[total_mask > 1] = 1

        #watershed_image = cv2.resize(self.data["hed"], (self.image.shape[1], self.image.shape[0]))
        denoised = rank.median(self.image[:,:,1], disk(2))
        watershed_image = rank.gradient(denoised, disk(2))

        watershed_mask = np.ones(watershed_image.shape, dtype=np.int32)
        markers = np.zeros(watershed_image.shape, dtype=np.int32)

        # print("Markers, image:", markers.shape, self.image.shape)

        def draw_surface_markers(surface, mask, color, freedom=0.15):
            if surface.surfaceType == SurfaceType.OnFloor:
                watershed_mask[mask > 0] = 0
            dist_transform = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)
            markers[dist_transform > freedom * dist_transform.max()] = color

        sy = self.data["image"].shape[0] / self.data["downscaled"].shape[0]
        sx = self.data["image"].shape[1] / self.data["downscaled"].shape[1]

        def draw_barrier_markers(barrier_groups):
            for barrier in barrier_groups:
                barrier.line.draw(markers, color= -1, thickness=1, lineType=cv2.LINE_4, sx=sx, sy=sy)
                barrier.line.draw(watershed_mask, color=0, thickness=1, lineType=cv2.LINE_4, sx=sx, sy=sy)

        num_surfaces = len(self.room.surfaces)
        for index in range(num_surfaces):
            surface = self.room.surfaces[index]
            draw_surface_markers(surface, mask=masks[index], color=index+1)

        for sb in self.barriers.values():
            draw_barrier_markers(sb.barrier_groups)
            
        markers[disputed_areas > 0] = 0

        log_markers(self.data, "room_markers", markers)

        markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))
        markers[markers<0] = 0

        log_markers(self.data, "room_markers_result", markers)

        #set masks:
        for index in range(num_surfaces):
            surface = self.room.surfaces[index]

            if surface.surfaceType == SurfaceType.OnFloor: 
                continue

            color = index + 1
            mask = np.zeros_like(total_mask)
            mask[markers == color] = 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            mask = cv2.dilate(mask, kernel)
            #mask[watershed_mask == 0] = 0
            surface.final_mask = mask

        log_image(self.data, "room_final", self.room.get_debug_image())
        
        
class PipelineSurfaceRefinement(PipelineStep):

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Refine

    @property
    def required_keys(self) -> list:
        return ["barriers"]

    @property
    def output_keys(self) -> list:
        return ["segmentation"]

    def run(self, data):
        refiner = SurfaceRefinement(data)
        refiner.refine()

        
