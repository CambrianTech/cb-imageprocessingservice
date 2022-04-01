import numpy as np
from scipy import ndimage
import cv2
from skimage.morphology import remove_small_objects

from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.misc.utils import resize_array, random_color, overlay_mask
from .planegeometry import Dimension
from pipeline.data.logging import log_image, log_segmentation_image, im_logging_enabled
from pipeline.components.line import Line
from pipeline.components.room import Room
from pipeline.components.surface import Surface

class RoomSolver():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.room = self.data["room"]

    def solve(self):
        
        #add all the applicable surfaces:
        self.room.analyze()

        return self.room


class PipelineRoomSolver(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.SolveRoom

    @property
    def required_keys(self) -> list:
        return ["room"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):
        solver = RoomSolver(data)
        data["room"] = solver.solve()

