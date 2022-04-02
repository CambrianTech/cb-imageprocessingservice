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

        def run_watershed(freedom):

            src = cv2.bilateralFilter(self.data["image"], 21, 80, 80)

            markers = np.zeros((src.shape[0], src.shape[1]), dtype=np.int32)

            def draw_surface_markers(surface, mask, color):

                timer.reset()
                dist_transform = surface.mask_transform

                #uncommon case where image was smaller than neural net size
                if dist_transform.shape[0] != src.shape[0] or dist_transform.shape[1] != src.shape[1]:
                    dist_transform = cv2.resize(dist_transform, (src.shape[1], src.shape[0])) 

                markers[dist_transform > freedom * dist_transform.max()] = color
                timer.time_event("dist_transform")

            for index in range(num_surfaces):
                surface = self.room.surfaces[index]
                timer.reset()
                if self.masks[index].shape[0] != src.shape[0] or self.masks[index].shape[1] != src.shape[1]:
                    self.masks[index] = cv2.resize(self.masks[index], (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST) 
                draw_surface_markers(surface, mask=self.masks[index], color=index+1)
                timer.time_event("draw_surface_markers")

            timer.reset()

            vp_lines = []

            if self.room.vertical_vp is not None and len(self.room.vertical_vp) > 0:
                vp_lines.extend(self.room.vertical_vp[0].inliers)

            for surface in self.room.surfaces:
                if surface.horizontal_vp is not None and len(surface.horizontal_vp) > 0:
                    vp_lines.extend(surface.horizontal_vp[0].inliers)

                if surface.vp and len(surface.vp):
                    vp_lines.extend(surface.vp[0].inliers)

            #lines = [line.extended(1.2) for line in lines]

            timer.reset()

            src_scale = src.shape[0] / self.data["downscaled"].shape[0]

            draw_lines(src, vp_lines, color=(0,0,0), scale=src_scale)
            draw_lines(src, self.data["lines"], color=(255,0,255), scale=src_scale)
            draw_lines(markers, vp_lines, color=255, scale=src_scale, lineType=cv2.LINE_4)

            log_markers(self.data, "room_markers", markers, primary=True)

            markers = cv2.watershed(src, markers)
            timer.time_event("cv2.watershed")

            markers[markers<0] = 0
            markers[markers==255] = 0

            log_image(self.data, "room_markers_src", src, primary=True)
            log_markers(self.data, "room_markers_result", markers, primary=True)

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

        #log_image(self.data, "room_refined_pre", self.room.get_debug_image(hires=True))

        #run_watershed(freedom=0.07)

        timer.log_all_events()

        for index in range(num_surfaces):
            surface = self.room.surfaces[index]
            surface.final_mask = self.masks[index]
        
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

        if "room" in data:
            data["room"] = refiner.room

        data["masks"] = refiner.masks

        
