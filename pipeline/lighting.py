from pipeline.core import PipelineStep
import cv2
import numpy as np
from scipy import ndimage
from cambrian import image_processing as ip
from skimage.morphology import watershed, disk
from skimage import filters
from skimage.filters import threshold_multiotsu


IM_LOGGING_ENABLED = False


def _log_image(name, image):
    if IM_LOGGING_ENABLED:
        cv2.imwrite(name, image)



class PipelineRefineLighting(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["mask", "lighting"]

    @property
    def output_keys(self) -> list:
        return ["lighting"]

    def run(self, data):
        mask = data["mask"]

        lighting_rgb = np.uint8(data["lighting"])
        lighting_rgb = cv2.resize(lighting_rgb, mask.shape)
        lighting = lighting_rgb[:, :, 1]

        # Remove hard edges from lighting
        smooth_lighting = ip.remove_grooves(lighting, mask)

        blurred_mask = cv2.GaussianBlur(mask, (31, 31), 15)
        blurred_lighting = cv2.GaussianBlur(smooth_lighting, (21, 21), 11)

        lighting = ip.alpha_blend(
            smooth_lighting, blurred_lighting, blurred_mask)

        data["lighting"] = lighting

        _log_image('lighting.png', lighting)