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
        num_surfaces = len(self.room.surfaces)

        #there's an issue with the shaders causing too close indices to mix up and creating artifacts
        #so giving spaced out indices randomly is a temporary solution
        
        indices = random.sample(range(1, 254), num_surfaces) #255 is off limits for internal use

        for i, surface in enumerate(self.room.surfaces):

            color = i + 1
            mask = np.zeros_like(surface.hires_mask)
            mask[markers == color] = 1
            mask[index_mask > 0] = 0

            surface.hires_mask = mask

            surface.index = indices[i]
            index_mask[surface.hires_mask > 0] = surface.index

            log_mask(self.data, "surface_%d" % surface.index, mask, background=self.image)


        #fill all gaps
        index_mask = np.uint8(watershed(watershed_image, index_mask))

        # #USE the alpha if you need transparency (below)

        # alpha = np.ones(self.image.shape[:2], dtype=np.uint8) * 255
        # for surface in self.room.surfaces:

        #     mask = np.zeros_like(surface.hires_mask)
        #     mask[index_mask == surface.index] = 1

        #     contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        #     cv2.drawContours(alpha, contours, -1, 220, 1, cv2.LINE_AA)

        # alpha = cv2.GaussianBlur(alpha,(13,13),0)

        # #integrate alpha value, convert to RGBA
        # index_mask = cv2.cvtColor(index_mask, cv2.COLOR_GRAY2RGB)
        # alpha = np.reshape(alpha, (*alpha.shape,1))
        # index_mask = np.concatenate([index_mask, alpha], axis=2)

        print("index shape", index_mask.shape)

        self.data["index_mask"] = index_mask
        log_image(self.data, "room_final", self.room.get_debug_image(hires=True)) 

        bw = cv2.cvtColor(self.data["downscaled"], cv2.COLOR_RGB2GRAY)
        bw = cv2.bilateralFilter(bw, d=15, sigmaColor=15, sigmaSpace=20)

        walls_mask = np.zeros(bw.shape[:2], dtype=np.uint8)
        walls_mask[self.data["isolated_labels"] == SurfaceType.Wall] = 1
        if cv2.countNonZero(walls_mask) < 500:
            walls_mask = None

        floor_mask = np.zeros(bw.shape[:2], dtype=np.uint8)
        floor_mask[self.data["isolated_labels"] == SurfaceType.Floor] = 1
        floor_mask[self.data["isolated_labels"] == SurfaceType.OnFloor] = 1
        if cv2.countNonZero(floor_mask) < 500:
            floor_mask = None

        lighting = self.data["lighting"].astype(np.uint8)
        lighting = cv2.resize(lighting, (bw.shape[1], bw.shape[0]))
        lighting = cv2.cvtColor(lighting, cv2.COLOR_RGB2GRAY)

        log_image(self.data, 'lighting', lighting)

        def scale_lighting(img, scale=1.0, center=127.0, gamma=10.0):

            img = (img.astype(float) - center) * scale + center + gamma

            img[img > 255.0] = 255.0
            img[img < 0.0] = 0.0
            return img.astype(np.uint8)
        
        # smooth_opacity = 0.3
        # gamma = 60
        # smoothed = cv2.pyrMeanShiftFiltering(scale_lighting(lighting), 5, 5)
        # lighting = cv2.addWeighted(smoothed, smooth_opacity, scale_lighting(lighting), 1 - smooth_opacity, gamma)

        mean, std = cv2.meanStdDev(lighting, mask=walls_mask)
        normal_luminance_std = 30.0
        max_scale = max(normal_luminance_std / (1 + std[0]), 0.5)
        lighting_scale = 1.3
        lighting_scale = min(max_scale, lighting_scale)
        center = 127.0 * lighting_scale

        #print("std", std[0])
        lighting = scale_lighting(lighting, scale=lighting_scale, center=center, gamma=100.0)
        lighting = cv2.bilateralFilter(lighting, d=15, sigmaColor=30, sigmaSpace=30)
        lighting_smoothed = lighting.copy()

        log_image(self.data, 'lighting_smooth', lighting)

        opacity = 0.8
        hed_weight = 0.15
        gamma = 0.0
        hed = cv2.resize(self.data["hed"], (bw.shape[1], bw.shape[0])).astype(float)
        hed = cv2.normalize(hed, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        #hed[bw > 200] = 0 #don't unlighten
        hed = cv2.medianBlur(hed, 5)
        lighting = lighting - hed * hed_weight * 255.0

        lighting[lighting < 0] = 0

        #merge lighting with background
        lighting = cv2.addWeighted(lighting.astype(np.uint8), opacity, bw, 1.0 - opacity, gamma)

        #handle floor separately
        if floor_mask is not None:
            floor_mask_blurred = cv2.blur(floor_mask.astype(float), (5, 5), 0)
            #log_image(self.data, 'floor_mask_blurred', floor_mask_blurred * 255)

            floor_lighting = lighting.copy()
            floor_lighting[floor_mask_blurred > 0] = lighting_smoothed[floor_mask_blurred > 0]

            #floor_lighting = scale_lighting(floor_lighting, scale=1.1, gamma=10.0) #gamma is scale_lighting above plus this

            lighting = floor_lighting * floor_mask_blurred + lighting * (1.0 - floor_mask_blurred)
        
        self.data["lighting"] = lighting

        self.data["room"].recalculate_lighting()

        log_image(self.data, 'lighting_final', self.data["lighting"])
        
        
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

        
