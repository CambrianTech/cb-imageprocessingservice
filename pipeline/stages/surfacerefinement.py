import numpy as np
from scipy import ndimage
import cv2

from skimage.morphology import skeletonize, remove_small_objects
from skimage.segmentation import join_segmentations, watershed

import cambrian.image_processing as ip

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.surface_type import SurfaceType
from pipeline.components.line import Line
from .planegeometry import Dimension
from pipeline.misc.utils import get_segmentation_image, random_color
from pipeline.data.logging import log_segmentation_image, im_logging_enabled, log_image, LogLevel

class SurfaceRefinement():
    def __init__(self, data):
        super().__init__()
        self.data = data
        
    def refine(self):
        self.data["lighting"] = cv2.edgePreservingFilter(np.uint8(self.data["lighting"]), flags=1, sigma_s=10, sigma_r=1.0)
        log_image(self.data, 'lighting_smooth', self.data["lighting"])
        
        
class PipelineSurfaceRefinement(PipelineStep):

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Refine

    @property
    def required_keys(self) -> list:
        return ["downscaled", "isolated", "lines", "hed"]

    @property
    def output_keys(self) -> list:
        return ["segmentation"]

    def run(self, data):
        refiner = SurfaceRefinement(data)
        refiner.refine()

        
