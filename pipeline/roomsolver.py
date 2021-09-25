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

            #for i, surface in enumerate(self.room.surfaces): log_image(self.data, "surface_%d" % i, surface.probs * 255)
            log_image(self.data, "room", self.room.get_debug_image())
            log_image(self.data, "room-unmasked", self.room.get_debug_image(False))


            log_segmentation_image(self.data, "room_masks", self.room.index_mask, self.room.image, show_legend=False)
            
            sx = self.room.image.shape[1] / self.room.data["image"].shape[1]
            sy = self.room.image.shape[0] / self.room.data["image"].shape[0]
            debug = log_segmentation_image(self.data, "probs", self.room.labels, self.room.image, get_image=True, labelset=SurfaceType)
            Line.draw_all(debug, self.room.lines, color=(255,80,200), thickness=2, sx=sx, sy=sy)
            log_image(self.data, "surfaces", debug)

            # for index in range(len(self.room.surfaces)):
            #     surface = self.room.surfaces[index]
            #     log_image(self.data, "skeleton_%d" % index, surface.skeleton * 255)
                

            


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
        solver.solve()

        # refiner = SurfaceRefinement(img_lr, hed_lr, data["isolated"], data["lines"])
        # data["segmentation"] = refiner.refine(data)

