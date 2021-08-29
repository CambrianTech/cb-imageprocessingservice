import cv2
import numpy as np

from pipeline.core import PipelineStep
from pipeline.logging import get_segmentation_image, log_image, log_segmentation_image, log_ply, im_logging_enabled, LogLevel

class PipelineRefineResults(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "hed", "normals"]

    @property
    def output_keys(self) -> list:
        return ["mask", "lighting"]

    def run(self, data):
        img = data["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[1] > 1024:
            img = cv2.resize(img, (1024, int(img.shape[0] / img.shape[1] * 1024)))

        log_image(data, "image", img)
