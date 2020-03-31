import requests
import numpy as np
import pickle
from time import time

from .core import PipelineStep


def _remote_plane_detect(address, data):
    input_dicts = [{
        "image": datum["image"],
        "camera": datum["camera"]
    } for datum in data]

    response_bytes = requests.post(address, data=pickle.dumps(input_dicts)).content
    return pickle.loads(response_bytes)


def _remote_networks(address, data):
    images = [datum["image"] for datum in data]
    response_bytes = requests.post(address, data=pickle.dumps(images)).content
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
        t = time()
        detections = _remote_plane_detect(self.address, data)
        print("Remote planes took %.2f seconds" % (time() - t))

        for datum, detection in zip(data, detections):
            datum["planes"] = detection


class PipelineRemoteNetworks(PipelineStep):
    def __init__(self, address: str):
        super().__init__()
        self.address = address

    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["unlit", "lighting"]

    @property
    def is_batched(self) -> bool:
        return True

    def run(self, data: dict) -> None:
        t = time()
        response_dict = _remote_networks(self.address, data)
        print("Remote networks took %.2f seconds" % (time() - t))

        for datum, unlit, lighting in zip(data, response_dict["unlit"], response_dict["lighting"]):
            datum["unlit"] = unlit
            datum["lighting"] = lighting