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


def combine_plane_masks(plane_masks: np.ndarray) -> np.ndarray:
    num_planes = len(plane_masks)

    # Cut off black bars that are present
    # from plane-rcnn.
    plane_masks = plane_masks[:, 80:-80]

    # Start at 1 with plane indices here.
    # Later we subtract 1 so that -1 means no plane, and
    # actual plane indices start at 0.
    plane_indices = np.arange(1, num_planes + 1)

    # Combine the separate plane masks into a single plane mask
    # with the plane indices as values. Also store the alpha
    # values of the masks by summing them.
    if plane_masks.dtype == np.uint8:
        plane_masks = plane_masks.astype(np.float32) / 255

    alpha_mask = np.sum(plane_masks, axis=0, dtype=np.float32)

    bool_masks = plane_masks.astype(np.bool).astype(np.float32)

    # [NumPlanes] @ [NumPlanes, H, W] => [H, W]
    # -1: No plane
    # range(nPlanes): plane index
    index_mask = np.einsum(
        "i,ihw->hw", plane_indices, bool_masks
    ).astype(np.int16) - 1

    return index_mask, alpha_mask


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

    def run(self, data: dict) -> None:
        t = time()
        plane_rcnn_outputs = _remote_plane_detect(self.address, data)
        print("Remote planes took %.2f seconds" % (time() - t))

        for datum, plane_rcnn_output in zip(data, plane_rcnn_outputs):
            index_mask, alpha_mask = combine_plane_masks(
                plane_rcnn_output["masks"]
            )

            datum["planes_index_mask"] = index_mask
            datum["planes_alpha_mask"] = alpha_mask
            datum["planes"] = plane_rcnn_output


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
