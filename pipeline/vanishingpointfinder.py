import numpy as np
from scipy import ndimage
import cv2
from skimage.morphology import remove_small_objects

from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .utils import resize_array, random_color, overlay_mask
from .planegeometry import Dimension
from .logging import log_image, log_segmentation_image, im_logging_enabled
from .Line import Line
from .room import Room, Surface

class VanishingPointFinder():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.room = data["room"]

    def solve(self, confidence=0.95):
        if self.room:
            print("find vanishing points for each surface")


class PipelineVanishingPointFinder(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.VanishingPoints

    @property
    def required_keys(self) -> list:
        return ["room", "downscaled", "isolated", "lines"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):

        VanishingPointFinder(data).solve()

