import requests
import numpy as np
import pickle
from time import time
import cv2

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.misc.utils import camera_fov_res_to_intrinsics

def _remote_plane_detect(address, data):
    print("Remote detection", address)
    input_dicts = [{
        "image": datum["image"],
        "camera": camera_fov_res_to_intrinsics(datum["fov"], np.array([datum["image"].shape[1], datum["image"].shape[0]], dtype=np.float32))[0]
    } for datum in data]

    response_bytes = requests.post(
        address, data=pickle.dumps(input_dicts)).content
    return pickle.loads(response_bytes)

class PipelinePlaneDetector(PipelineStep):

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.PlaneDetector

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
        print("Running remote planes", self.config.planes_url)
        plane_rcnn_outputs = _remote_plane_detect(self.config.planes_url, data)
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

