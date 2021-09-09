import requests
import numpy as np
import pickle
from time import time
import cv2

from .core import PipelineStep, PipelineStepIndex
from .utils import camera_fov_res_to_intrinsics

def _remote_plane_detect(address, data):
    input_dicts = [{
        "image": datum["image"],
        "camera": camera_fov_res_to_intrinsics(datum["fov"], np.array([datum["image"].shape[1], datum["image"].shape[0]], dtype=np.float32))[0]
    } for datum in data]

    response_bytes = requests.post(
        address, data=pickle.dumps(input_dicts)).content
    return pickle.loads(response_bytes)

def _remote_networks(address, data):
    images = [datum["image"] for datum in data]
    response_bytes = requests.post(address, data=pickle.dumps(images)).content
    return pickle.loads(response_bytes)


class PipelineRemotePlaneDetector(PipelineStep):

    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.RemotePlaneDetector

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
        plane_rcnn_outputs = _remote_plane_detect(self.pipeline.planes_url, data)
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

    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.RemoteNetworks

    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["lighting", "normals"]

    @property
    def is_batched(self) -> bool:
        return True

    def run(self, data: dict) -> None:
        t = time()
        response_dict = _remote_networks(self.pipeline.remote_path, data)
        print("Remote networks took %.2f seconds" % (time() - t))

        for datum, lighting, normals in zip(data, response_dict["lighting"], response_dict["normals"]):
            datum["lighting"] = lighting
            datum["normals"] = normals

