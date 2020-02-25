import torch
from planercnn.options import parse_args
from planercnn.evaluate import PlaneRCNNDetector
from planercnn.config import InferenceConfig

from .core import PipelineStep

class PipelineRunPlaneNetwork(PipelineStep):
    def __init__(self):
        super().__init__()

        options = parse_args()
        config = InferenceConfig(options)
        self.detector = PlaneRCNNDetector(options, config, modelType="final")

    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["planes"]

    def run(self, data):
        # TODO: Enable batching

        image = data["image"]
        image_meta = [
            0, # Image id
            0, 0, 0, 0, # Image shape
            0, 0, 0, 0, # Window
            # Active class ids
        ]

        sample = [
            image,
            image_meta
        ]

        with torch.no_grad():
            detections = self.detector.detect(sample)
            data["planes"] = detections