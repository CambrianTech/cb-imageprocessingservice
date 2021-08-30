import cv2
import numpy as np
import random

from pipeline.core import PipelineStep
from pipeline.logging import get_segmentation_image, log_image, log_segmentation_image, log_ply, im_logging_enabled, LogLevel
from pipeline.ade20k import ADE20K
from pipeline.semantics import combine_floor_masks, isolate_masks, Groupings
from cambrian.VanishingPointFinder import VanishingPointFinder
from pipeline.fovestimator import calcPlaneXYZ, FovEstimator

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

class PipelineRefineResults(PipelineStep):

    def __init__(self):
        super().__init__()

    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "lines", "isolated"]

    @property
    def output_keys(self) -> list:
        return ["mask", "lighting"]


    def run(self, data):
        self.img = data["image"]

        self.output = data["output"]

        self.height, self.width = self.img.shape[:2]
        self.diagonal = np.hypot(self.width, self.height)

        img_lr = data["downscaled"] if "downscaled" in data else self.img
        shape = (img_lr.shape[1], img_lr.shape[0])

        self.isolated = data["isolated"]

        #calculate fov and surface vanishing points:
        fov_estimator = FovEstimator(data, self.img, data["lines"], data["fov"], self.isolated[Groupings.Floor], data["floor_normal"], data["floor_offset"])
        fov_estimator.estimate(shape)

        #vanishing points, may not be present!:
        
                
