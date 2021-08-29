from pipeline.core import PipelineStep
import cv2
import numpy as np
from scipy import ndimage
from cambrian import image_processing as ip
from skimage.morphology import watershed, disk
from skimage import filters
from skimage.filters import threshold_multiotsu


IM_LOGGING_ENABLED = False


def _log_image(name, image):
    if IM_LOGGING_ENABLED:
        cv2.imwrite('logging/' + name, image)



class PipelineRefineResults(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "hed", "normals"]

    @property
    def output_keys(self) -> list:
        return ["mask", "lighting"]

    def run(self, data):
        
        print("Refinement!")
