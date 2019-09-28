from pipeline.core import PipelineStep

import numpy as np
import cv2
import numba


@numba.njit
def get_label_bounding_boxes(label_image):
    """Find the min and max x and y coordinates for each label"""
    label_bounds = {}

    for x in range(label_image.shape[0]):
        for y in range(label_image.shape[1]):
            label = label_image[x, y]
            if label not in label_bounds:
                label_bounds[label] = np.array([x, y, x, y])

            bounds = label_bounds[label]
            bounds[0] = min(bounds[0], x)
            bounds[1] = min(bounds[1], y)
            bounds[2] = max(bounds[2], x)
            bounds[3] = max(bounds[3], y)

    return label_bounds


@numba.njit
def get_patch_with_label(label_image, bounds, label):
    return (label_image[bounds[0]:bounds[2], bounds[1]:bounds[3]] == label).astype(np.uint8)


@numba.jit(forceobj=True)
def get_label_contours(label_image, label_bounds):
    contours = {}
    for label, bounds in label_bounds.items():
        patch = get_patch_with_label(label_image, bounds, label)
        label_contours, _ = cv2.findContours(
            patch, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(label_contours) > 0:
            contours[label] = (
                label_contours[0][:, 0, :] + bounds[:2]).tolist()
    return contours


class PipelineSuperpixels(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["superpixel_image", "superpixel_contours"]

    def run(self, data):
        image = cv2.resize(data["image"], (1024, 1024))

        superpixels = cv2.ximgproc.createSuperpixelLSC(image)
        superpixels.iterate()
        superpixels.enforceLabelConnectivity()

        superpixel_image = superpixels.getLabels()

        label_bounds = get_label_bounding_boxes(superpixel_image)
        label_contours = get_label_contours(superpixel_image, label_bounds)

        # Convert 32 bit to 4x 8bit (https://stackoverflow.com/a/25298780)
        bytes_dtype = np.dtype(("i4", [("bytes", "u1", 4)]))
        superpixel_image = superpixel_image.view(dtype=bytes_dtype)["bytes"]

        # Invert alpha channel. This serves no purpose other than visualization
        # as otherwise all pixels would be transparent.
        superpixel_image[:, :, 3] = 255 - superpixel_image[:, :, 3]

        data["superpixel_contours"] = label_contours
        data["superpixel_image"] = superpixel_image
