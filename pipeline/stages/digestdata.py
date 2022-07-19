import numpy as np

import cv2
import random
import time
import math
from scipy.spatial import distance
from operator import attrgetter
from skimage.segmentation import watershed
import itertools

from cambrian.LineFunctions import LineFunctions

from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_image, im_logging_enabled, log_markers, log_segmentation_image, log_mask
from pipeline.components.line import draw_lines
from .vanishingpointfinder import angle_with_vp
from pipeline.misc.utils import random_color
from pipeline.data.ade20k import ADE20K
from pipeline.stages.extractsurfaces import on_floor, on_wall, on_ceiling, box_like, legged_objects
from pipeline.components.rotated_rect import RotatedRect

class PipelineDigestData(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.DigestData

    @property
    def required_keys(self) -> list:
        return ["room"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):

        data["lighting"] = cv2.edgePreservingFilter(np.uint8(data["lighting"]), flags=1, sigma_s=10, sigma_r=1.0)
        log_image(data, 'lighting_smooth', data["lighting"])

        #Consolidate types: Include other types as part of floor: rug, earth, grass
        output = data["semantic_probs"]

        h, w = output[0].shape
        shape = (w, h)

        log_image(data, "hed", data["hed"])

        if data["image"].shape[0] > shape[0] or data["image"].shape[1] > shape[1]:
            data["downscaled"] = cv2.resize(data["image"], shape)
        else:
            data["downscaled"] = data["image"]


        data["semantic_labels"] = np.argmax(np.dstack(data["semantic_probs"]), -1)

