import requests
import numpy as np
import pickle
from time import time
import cv2

from .core import PipelineStep


def _camera_fov_res_to_intrinsics(fov: float, res: np.ndarray):
    # https://stackoverflow.com/a/41137160
    # fov = 2 * arctan(r / (2 * f)) <=> f_y = r / (2 * tan(fov / 2))
    c = res / 2

    # Assume the fov corresponds to the longest side and use that for focal
    i = 0 if c[0] >= c[1] else 1
    f = c[i] / np.tan(np.radians(fov) / 2)

    return np.array([f, f, c[0], c[1], res[0], res[1]], dtype=np.float32)


def _remote_plane_detect(address, data):
    input_dicts = [{
        "image": datum["image"],
        "camera": _camera_fov_res_to_intrinsics(datum["fov"], np.array([datum["image"].shape[0], datum["image"].shape[1]], dtype=np.float32))
    } for datum in data]

    response_bytes = requests.post(
        address, data=pickle.dumps(input_dicts)).content
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
        return ["image", "fov"]

    @property
    def output_keys(self) -> list:
        return ["planes"]

    @property
    def is_batched(self) -> bool:
        return True

    def run(self, data):
        t = time()
        plane_rcnn_outputs = _remote_plane_detect(self.address, data)
        print("Remote planes took %.2f seconds" % (time() - t))

        for datum, plane_rcnn_output in zip(data, plane_rcnn_outputs):
            datum["planes"] = plane_rcnn_output

            # Plane-rcnn originally added black bars on top and bottom.
            # We cut those out so we need to subtract 80
            # (= (640 - 480) / 2) from the Y coordinates.

            # Plane masks
            datum["planes"]["masks"] = datum["planes"]["masks"][:, 80:-80]

            # Extents
            datum["planes"]["detection"][:, 0] -= 80  # min y
            datum["planes"]["detection"][:, 2] -= 80  # max y


class PipelineRemoteNetworks(PipelineStep):
    def __init__(self, address: str):
        super().__init__()
        self.address = address

    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["unlit", "lighting", "normals"]

    @property
    def is_batched(self) -> bool:
        return True

    def run(self, data: dict) -> None:
        t = time()
        response_dict = _remote_networks(self.address, data)
        print("Remote networks took %.2f seconds" % (time() - t))

        for datum, unlit, lighting, normals in zip(data, response_dict["unlit"], response_dict["lighting"], response_dict["normals"]):
            datum["unlit"] = unlit
            datum["lighting"] = lighting
            datum["normals"] = normals

