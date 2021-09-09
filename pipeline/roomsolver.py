import numpy as np
from scipy import ndimage
import cv2

from .core import PipelineStep, PipelineStepIndex
from .utils import resize_array, random_color
from .planegeometry import Dimension
from .logging import log_image, log_segmentation_image, im_logging_enabled
from .Line import Line
from .extractsurfaces import Groupings

class RoomSolver():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.image = self.data["downscaled"]

        planes_data = self.data["planes"]
        shape = (self.image.shape[1], self.image.shape[0])

        self.probs = self.data["isolated"]
        self.lines = self.data["lines"]
        self.plane_masks = resize_array(planes_data["masks"], shape)
        self.vert_indices = self.data["dimensions"][Dimension.Vertical].indices


    def solve(self, confidence=0.95):
        
        wall_contours = []

        for i in self.vert_indices:
            mask = self.plane_masks[i] * 255
            mask[mask < 127] = 0
            log_image(self.data, "mask_%d" % i, mask)
            wall_contours.append(cv2.findContours(np.uint8(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE))


        items = self.probs.copy()
        items.insert(0, (confidence * np.ones_like(self.probs[Groupings.Other])))

        #take intersection
        ade_seg_c = np.dstack(tuple(items))
        ade_seg = np.argmax(ade_seg_c, -1)
        
        sx = self.image.shape[1] / self.data["image"].shape[1]
        sy = self.image.shape[0] / self.data["image"].shape[0]
        
        if im_logging_enabled(self.data):
            debug = log_segmentation_image(self.data, "probs", np.int32(ade_seg), self.image, get_image=True)

            for contours, hierarchy in wall_contours:
                color = random_color()
                cv2.drawContours(debug, contours, -1, color, 2)

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

