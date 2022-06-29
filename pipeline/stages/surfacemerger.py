import numpy as np
from scipy import ndimage
import cv2
from skimage.morphology import remove_small_objects

from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_image, log_segmentation_image, im_logging_enabled, Timer
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

        print("MERGE")
        for surface in data["room"].surfaces:
            print("surface")