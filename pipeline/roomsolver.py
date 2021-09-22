import numpy as np
from scipy import ndimage
import cv2
from skimage.morphology import remove_small_objects

from .core import PipelineStep, PipelineStepIndex
from .utils import resize_array, random_color, overlay_mask
from .planegeometry import Dimension
from .logging import log_image, log_segmentation_image, im_logging_enabled
from .Line import Line
from .extractsurfaces import Groupings
from .room import Room, Surface

class RoomSolver():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.probs = self.data["isolated"]
        self.lines = self.data["lines"]

        self.room = Room(data)

    def solve(self, confidence=0.95):
        
        wall_contours = []

        self.image = self.data["downscaled"]
        sx = self.image.shape[1] / self.data["image"].shape[1]
        sy = self.image.shape[0] / self.data["image"].shape[0]
        
        
        #Get argmax:

        #take intersection
        #self.probs.insert(0, (confidence * np.ones_like(self.probs[Groupings.Other])))
        ade_seg_c = np.dstack(tuple(self.probs))
        ade_seg = np.argmax(ade_seg_c, -1)

        labels = np.int32(ade_seg)

        #add the walls:
        for i in range(len(self.room.masks)):
            self.room.add_surface(Surface(self.data, i))

        self.room.analyze(labels)
        
        if im_logging_enabled(self.data):

            #for i, surface in enumerate(self.room.surfaces): log_image(self.data, "surface_%d" % i, surface.probs * 255)
            log_image(self.data, "room", self.room.get_debug_image())

            
            # for index in range(len(self.room.surfaces)):
            #     surface = self.room.surfaces[index]
            #     log_image(self.data, "skeleton_%d" % index, surface.skeleton * 255)
                

            debug = log_segmentation_image(self.data, "probs", np.int32(ade_seg), self.image, get_image=True, labelset=Groupings)

            Line.draw_all(debug, self.lines, color=(255,80,200), thickness=2, sx=sx, sy=sy)

            log_image(self.data, "surfaces", debug)


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

