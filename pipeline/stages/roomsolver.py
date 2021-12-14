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
        self.room = Room(data)

    def solve(self, confidence=0.95):
        
        wall_contours = []

        #add the walls:
        for i in range(len(self.room.probs)):
            self.room.add_surface(Surface(self.data, i))

        self.room.analyze()

        if im_logging_enabled(self.data):

            log_segmentation_image(self.data, "room_masks", self.room.index_mask, self.room.image)
            
            debug = log_segmentation_image(self.data, "isolated_labels", self.room.isolated_labels, self.room.image, get_image=True, labelset=SurfaceType)

            log_image(self.data, "surfaces", debug)


        return self.room
            


class PipelineRoomSolver(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.SolveRoom

    @property
    def required_keys(self) -> list:
        return ["planes", "downscaled", "isolated", "lines", "dimensions"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):

        solver = RoomSolver(data)
        data["room"] = solver.solve()

