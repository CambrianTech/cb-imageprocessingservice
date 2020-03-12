import requests
import numpy as np
import pickle

from .core import PipelineStep


def _remote_plane_detect(address, data):
    input_dicts = [{
        "image": datum["image"],
        "camera": datum["camera"]
    } for datum in data]

    response_bytes = requests.post(address, data=pickle.dumps(input_dicts)).content
    return pickle.loads(response_bytes)


class PipelineRemotePlaneDetector(PipelineStep):
    def __init__(self, address: str):
        super().__init__()
        self.address = address

    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["planes"]

    @property
    def is_batched(self) -> bool:
        return True

    def run(self, data: dict) -> None:
        detections = _remote_plane_detect(self.address, data)

        for datum, detection in zip(data, detections):
            datum["planes"] = detection