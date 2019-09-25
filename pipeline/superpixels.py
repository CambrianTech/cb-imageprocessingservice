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

        superpixels = cv2.ximgproc.createSuperpixelLSC(image)
        superpixels.iterate()
        superpixels.enforceLabelConnectivity()

        superpixel_image = superpixels.getLabels()

        # Convert 32 bit to 4x 8bit (https://stackoverflow.com/a/25298780)
        bytes_dtype = np.dtype(("i4", [("bytes", "u1", 4)]))
        superpixel_image = superpixel_image.view(dtype=bytes_dtype)["bytes"]

        # Invert alpha channel. This serves no purpose other than visualization
        # as otherwise all pixels would be transparent.
        superpixel_image[:, :, 3] = 255 - superpixel_image[:, :, 3]

        data["superpixels"] = superpixel_image
