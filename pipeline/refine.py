import cv2
import numpy as np

from pipeline.core import PipelineStep
from pipeline.logging import get_segmentation_image, log_image, log_segmentation_image, log_ply, im_logging_enabled, LogLevel
from pipeline.ade20k import ADE20K
from pipeline.semantics import combine_floor_masks, isolate_masks, Groupings
from pipeline.linefinder import LineFinder

class PipelineRefineResults(PipelineStep):

    def __init__(self, max_size=1024):
        super().__init__()
        self.max_size = max_size

    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "hed", "normals"]

    @property
    def output_keys(self) -> list:
        return ["mask", "lighting"]


    def run(self, data):
        self.img = data["image"]

        if self.img.shape[1] > self.max_size:
            self.img = cv2.resize(self.img, (self.max_size, int(self.img.shape[0] / self.img.shape[1] * self.max_size)))

        log_image(data, "image", self.img)

        self.bw = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)

        self.output = np.float32(data["semantic_probs"])
        combine_floor_masks(self.output) #Include other types as part of floor: rug, earth, grass:

        self.height, self.width = self.img.shape[:2]
        self.diagonal = np.hypot(self.width, self.height)

        self.isolated = isolate_masks(data, self.output) #break masks into major groups: Floor, Wall, Ceiling, etc

        line_finder = LineFinder(self.img, self.bw)
        line_finder.detect(data)
