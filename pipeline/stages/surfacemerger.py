import numpy as np
from scipy import ndimage
import cv2
from skimage.morphology import remove_small_objects

from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_image, log_mask, log_segmentation_image, im_logging_enabled, Timer
from pipeline.components.scene import Scene
from pipeline.components.surface import Surface


class PipelineSurfaceMerger(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.SurfaceMerger

    @property
    def required_keys(self) -> list:
        return ["planes", "downscaled", "isolated", "dimensions"]

    @property
    def output_keys(self) -> list:
        return ["room"]

    def run(self, data):

        print("MERGE walls")

        for surface in data["room"].get_surfaces([SurfaceType.Wall]):
            
            # log_mask(data, surface.name + "_mask", surface.mask)
            # log_mask(data, surface.name + "_mask_edges", surface.mask_edges)
            # log_mask(data, surface.name + "_mask_expanded", surface.mask_expanded)

            smaller_neighbors = list(filter(lambda s: s.surfaceType == surface.surfaceType and s.max_area < surface.max_area, surface.neighbors))

            if len(smaller_neighbors) == 0: continue

            print("surface %s has smaller neighbors" % surface.name, [neighbor.name for neighbor in smaller_neighbors])

