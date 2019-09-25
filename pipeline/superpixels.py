from pipeline.core import PipelineStep

import numpy as np
import cv2


class PipelineSuperpixels(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["superpixels"]

    def run(self, data):
        image = data["image"]
        superpixels = cv2.ximgproc.createSuperpixelSLIC(image)
        superpixels.iterate(10)
        data["superpixels"] = superpixels.getLabels()
