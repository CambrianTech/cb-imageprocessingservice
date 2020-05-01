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


def get_ply(image, masks, plane_xyz, write_occlusion=False):
    width = image.shape[1]
    height = image.shape[0]

    faces = []
    points = []

    masks = np.round(masks)
    plane_depths = plane_xyz[:, :, :, 1] * masks + 10 * (1 - masks)
    segmentation = plane_depths.argmin(0)

    for mask_index, (mask, XYZ) in enumerate(zip(masks, plane_xyz)):
        indices = np.nonzero(mask > 0.5)
        for y, x in zip(indices[0], indices[1]):
            if y == height - 1 or x == width - 1:
                continue
            valid_neighbor_pixels = []
            for neighbor_pixel in [(x, y + 1), (x + 1, y), (x + 1, y + 1)]:
                if mask[neighbor_pixel[1], neighbor_pixel[0]] > 0.5:
                    valid_neighbor_pixels.append(neighbor_pixel)
                continue
            if len(valid_neighbor_pixels) == 3:
                faces.append([len(points) + c for c in range(3)])
                points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]]
                            == mask_index) for pixel in [(x, y), (x + 1, y + 1), (x + 1, y)]]
                faces.append([len(points) + c for c in range(3)])
                points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]]
                            == mask_index) for pixel in [(x, y), (x, y + 1), (x + 1, y + 1)]]
            elif len(valid_neighbor_pixels) == 2:
                faces.append([len(points) + c for c in range(3)])
                points += [
                    (XYZ[pixel[1], pixel[0]], pixel,
                     segmentation[pixel[1], pixel[0]] == mask_index)
                    for pixel in [
                        (x, y),
                        (valid_neighbor_pixels[0][0],
                         valid_neighbor_pixels[0][1]),
                        (valid_neighbor_pixels[1][0],
                         valid_neighbor_pixels[1][1])
                    ]
                ]

    ply_text = ""

    # Write header
    header = """ply
format ascii 1.0"""
    header += "textureless"
    header += """
element vertex """
    header += str(len(points))
    header += """
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
element face """
    header += str(len(faces))
    header += """
property list uchar int vertex_indices
end_header
"""
    ply_text += header

    # Write vertices
    for point in points:
        X = point[0][0]
        Y = point[0][1]
        Z = point[0][2]
        if not write_occlusion or point[2]:
            color = image[point[1][1], point[1][0]]
        else:
            color = (128, 128, 128)
        ply_text += str(X) + ' ' + str(Z) + ' ' + str(-Y) + ' ' + \
            str(color[2]) + ' ' + str(color[1]) + ' ' + str(color[0]) + '\n'

    # Write faces
    for face in faces:
        ply_text += '3 '
        for c in face:
            ply_text += str(c) + ' '
        ply_text += '\n'

    return ply_text


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
        detections = _remote_plane_detect(self.address, data)
        print("Remote planes took %.2f seconds" % (time() - t))

        for datum, detection in zip(data, detections):
            detection["ply"] = get_ply(
                cv2.resize(datum["image"], (640, 480)),
                detection["masks"][80:-80],
                np.transpose(detection["plane_XYZ"],
                             (0, 2, 3, 1))[:, 80:-80]
            )

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
