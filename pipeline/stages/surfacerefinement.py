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
from pipeline.misc.utils import get_segmentation_image, random_color
from pipeline.data.logging import log_segmentation_image, im_logging_enabled, log_image, LogLevel, log_markers, Timer

class SurfaceRefinement():
    def __init__(self, data):
        super().__init__()
        self.data = data

        self.room = self.data["room"]
        self.masks = [s.mask for s in self.room.surfaces]
        self.image = self.data["image"]
        self.barriers = self.data["barriers"] if "barriers" in self.data else None
        
    def refine(self, use_HED=False):
        self.data["lighting"] = cv2.edgePreservingFilter(np.uint8(self.data["lighting"]), flags=1, sigma_s=10, sigma_r=1.0)
        log_image(self.data, 'lighting_smooth', self.data["lighting"])

        num_surfaces = len(self.room.surfaces)

        timer = Timer("refine")
        #timer.disable()

        # refine.dist_transform took a total of 0.2879 seconds for 18 iterations, avg: 0.0160
        # refine.draw_surface_markers took a total of 0.0168 seconds for 18 iterations, avg: 0.0009
        # refine.draw_lines took a total of 0.0008 seconds for 1 iterations, avg: 0.0008
        # refine.cv2.watershed took a total of 0.0255 seconds for 1 iterations, avg: 0.0255
        # refine.set masks took a total of 0.0905 seconds for 1 iterations, avg: 0.0905

        def run_watershed(src, freedom=0.15, use_cv=False):

            watershed_mask = None if use_cv else np.ones(src.shape, dtype=np.int32)
            markers = np.zeros((src.shape[0], src.shape[1]), dtype=np.int32)

            def draw_surface_markers(surface, mask, color):
                if watershed_mask is not None and surface.surfaceType == SurfaceType.OnFloor:
                    watershed_mask[mask > 0] = 0

                timer.reset()
                #dist_transform = cv2.distanceTransform(mask, distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)
                dist_transform = surface.mask_transform

                #uncommon case where image was smaller than neural net size
                if dist_transform.shape[0] != src.shape[0] or dist_transform.shape[1] != src.shape[1]:
                    dist_transform = cv2.resize(dist_transform, (src.shape[1], src.shape[0])) 

                markers[dist_transform > freedom * dist_transform.max()] = color
                timer.time_event("dist_transform")

            sy = src.shape[0] / self.data["downscaled"].shape[0]
            sx = src.shape[1] / self.data["downscaled"].shape[1]

            def draw_barrier_markers(barrier_groups):
                for barrier in barrier_groups:
                    barrier.line.draw(markers, color= -1, thickness=1, lineType=cv2.LINE_4, sx=sx, sy=sy)
                    if watershed_mask is not None:
                        barrier.line.draw(watershed_mask, color=0, thickness=1, lineType=cv2.LINE_4, sx=sx, sy=sy)

            for index in range(num_surfaces):
                surface = self.room.surfaces[index]
                timer.reset()
                if self.masks[index].shape[0] != src.shape[0] or self.masks[index].shape[1] != src.shape[1]:
                    self.masks[index] = cv2.resize(self.masks[index], (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST) 
                draw_surface_markers(surface, mask=self.masks[index], color=index+1)
                timer.time_event("draw_surface_markers")

            timer.reset()
            if self.barriers is not None:
                #lines_mask = None
                for sb in self.barriers.values():
                    draw_barrier_markers(sb.barrier_groups)
                timer.time_event("draw_barrier_markers")

            elif use_cv:
                #bright green
                draw_lines(src, self.data["lines"], color=(0,255,0), sx=sx, sy=sy)
                timer.time_event("draw_lines")

            log_markers(self.data, "room_markers", markers)

            timer.reset()
            if use_cv:
                markers = cv2.watershed(src, markers)
                timer.time_event("cv2.watershed")
            else:
                markers = np.int32(watershed(src, markers, mask=watershed_mask))
                timer.time_event("watershed")
            markers[markers<0] = 0

            log_markers(self.data, "room_markers_result", markers)

            timer.reset()

            #set masks:
            for index in range(num_surfaces):
                surface = self.room.surfaces[index]

                if surface.surfaceType == SurfaceType.OnFloor: 
                    continue

                color = index + 1
                mask = np.zeros_like(self.masks[index])
                mask[markers == color] = 1
                # if lines_mask is not None:
                #     mask[lines_mask > 0] = 0

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
                mask = cv2.dilate(mask, kernel)
                #mask[watershed_mask == 0] = 0
                self.masks[index] = mask


            timer.time_event("set masks")

        # hed = cv2.resize(self.data["hed"], (self.image.shape[1], self.image.shape[0]))
        # run_watershed(hed, freedom=0.03)

        #denoised = rank.median(self.image[:,:,1], disk(5))
        #denoised = cv2.bilateralFilter(self.image[:,:,1], 9, 20, 20)
        run_watershed(self.image, freedom=0.05, use_cv=True)

        timer.log_all_events()

        for index in range(num_surfaces):
            surface = self.room.surfaces[index]
            surface.final_mask = self.masks[index]
        
        log_image(self.data, "room_final", self.room.get_debug_image())
            
        
        
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

        if "room" in data:
            data["room"] = refiner.room

        data["masks"] = refiner.masks

        
