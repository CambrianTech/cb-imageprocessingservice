import cv2
import numpy as np
import random

from pipeline.core import PipelineStep
from pipeline.logging import get_segmentation_image, log_image, log_segmentation_image, log_ply, im_logging_enabled, LogLevel
from pipeline.ade20k import ADE20K
from pipeline.extractsurfaces import Groupings
from cambrian.VanishingPointFinder import VanishingPointFinder
from pipeline.poseestimator import calcPlaneXYZ, fan_surfaces

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

class PipelineRefineResults(PipelineStep):

    def __init__(self):
        super().__init__()

    @property
    def required_keys(self) -> list:
        return ["image", "output", "lines", "isolated", "segmentation"]

    @property
    def output_keys(self) -> list:
        return ["mask", "lighting"]


    def run(self, data):
        self.img = data["image"]

        self.output = data["output"]

        self.height, self.width = self.img.shape[:2]
        self.diagonal = np.hypot(self.width, self.height)

        #process lighting
        lighting_rgb = np.uint8(data["lighting"])
        lighting_smooth = cv2.edgePreservingFilter(lighting_rgb, flags=1, sigma_s=10, sigma_r=1.0)
        log_image(data, 'lighting_smooth', lighting_smooth)
        data["lighting"] = lighting_smooth
        log_image(data, 'lighting', lighting_smooth)

        #finalize angles
        isolated = data["isolated"]
        img_lr = data["downscaled"]
        segmentation_initial = data["segmentation"]
        sure_floor = (segmentation_initial == ADE20K.floor.index)


        labels_fan, fan_normals_reduced, normals_wall = fan_surfaces(data, img_lr, data["edgelets"][0], data["vp0"], sure_floor, isolated[Groupings.Wall], data["normals_c"])

        
                
