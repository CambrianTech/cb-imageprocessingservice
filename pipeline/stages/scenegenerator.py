import numpy as np
from scipy import ndimage
import cv2
from skimage.morphology import remove_small_objects

from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_image, log_segmentation_image, im_logging_enabled, Timer
from pipeline.components.room import Room
from pipeline.components.surface import Surface

class SceneGenerator():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.room = Room(data)

    def generate(self):
        
        timer = Timer("room")

        #add all the applicable surfaces:
        for i in range(len(self.room.probs)):
            surface = Surface(self.data, i)
            self.room.add_surface(surface)
            surface.analyze()

        log_image(self.data, "room_analyzed", self.room.get_debug_image())

        timer.time_event("analyze_surfaces")

        if im_logging_enabled(self.data):

            log_segmentation_image(self.data, "room_masks", self.room.index_mask, self.room.image, labelset=None)
            log_segmentation_image(self.data, "isolated_labels", self.room.isolated_labels, self.room.image, labelset=SurfaceType, opacity=0.9)


        return self.room
            

class PipelineSceneGenerator(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.GenerateScene

    @property
    def required_keys(self) -> list:
        return ["planes", "downscaled", "isolated", "dimensions"]

    @property
    def output_keys(self) -> list:
        return ["room"]

    def run(self, data):

        solver = SceneGenerator(data)
        data["room"] = solver.generate()