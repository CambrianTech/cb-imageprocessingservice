import cv2
import numpy as np
import random
from skimage.morphology import remove_small_objects

import cambrian.image_processing as ip

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.logging import get_segmentation_image, log_image, log_segmentation_image, log_ply, im_logging_enabled, LogLevel
from pipeline.ade20k import ADE20K
from pipeline.extractsurfaces import Groupings
from cambrian.VanishingPointFinder import VanishingPointFinder
from pipeline.poseestimator import calcPlaneXYZ, fan_surfaces
from pipeline.utils import resize_array


def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

class PipelineRefineResults(PipelineStep):

    def __init__(self, pipeline, mask_size=2048):
        super().__init__(pipeline)
        self.mask_size = mask_size

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Refine

    @property
    def required_keys(self) -> list:
        return ["image", "output", "lines", "isolated", "segmentation"]

    @property
    def output_keys(self) -> list:
        return ["mask", "lighting"]

    def transfer_labels(self, data, final_labels, final_plane_number, final_plane_parameters, final_rotations):
        data["planes"]["masks"] = np.zeros((final_plane_number, final_labels.shape[0], final_labels.shape[1]),
                                           dtype=np.uint8)

        data["planes"]["detection"] = np.zeros((final_plane_number, 11), dtype=data["planes"]["detection"].dtype)
        data["planes"]["detection"] = final_plane_parameters
        data["planes"]["rotation"] = final_rotations

        for d in range(final_plane_number):
            contours, hierarchy = cv2.findContours(np.uint8(final_labels == d + 2), cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            data["planes"]["masks"][d] = np.zeros_like(np.uint8(final_labels == d + 2))

            if len(contours) > 0:

                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])

                    if area > 4 * 16 * 16:
                        if hierarchy[0, i, 3] == -1:  # this is the outer contour which we need to draw
                            cv2.drawContours(data["planes"]["masks"][d], [contours[i]], -1, 255, -1, cv2.LINE_AA)
                        else:
                            cv2.drawContours(data["planes"]["masks"][d], contours, i, 0, -1)

                data["planes"]["detection"][d, 0:4] = [0, 0, 0, 0]


    def run(self, data):
        self.img = data["image"]

        self.output = data["output"]

        self.height, self.width = self.img.shape[:2]
        self.diagonal = np.hypot(self.width, self.height)

        #process lighting
        lighting_rgb = np.uint8(data["lighting"])
        lighting_smooth = cv2.edgePreservingFilter(lighting_rgb, flags=1, sigma_s=10, sigma_r=1.0)
        log_image(data, 'lighting_smooth', lighting_smooth)
        data["lighting"] = lighting_smooth
        log_image(data, 'lighting', lighting_smooth)

        final_labels = data["final_labels"]
        final_masks = data["final_masks"]
        final_plane_parameters = data["plane_parameters"]
        final_rotations = data["plane_rotations"]

        img = data["image"]
        
        #scale to mask_size, watershed at that size:
        mask_shape = (self.mask_size , int(self.mask_size  * img.shape[0] / img.shape[1]))

        if img.shape[0] > img.shape[1]:
            mask_shape = (int(self.mask_size  * img.shape[1] / img.shape[0]), self.mask_size)

        final_masks_hr = resize_array(np.uint8(final_masks), mask_shape)
        final_labels_hr = np.int32(np.argmax(final_masks_hr, 0))

        final_labels_hr[final_labels_hr > 0] += 1
        final_labels_hr[final_masks_hr[0] > 0] = 1
        final_labels_hr += 1

        final_labels_hr = np.uint8(final_labels_hr)

        #todo: draw barriers such as lines

        final_labels_hr = ip.refine_mask_watershed(None, cv2.resize(img, (final_labels_hr.shape[1], final_labels_hr.shape[0])),
                                                   final_labels_hr, None, distance=0.01) - 1

        for i in np.unique(final_labels_hr):
            mask = final_labels_hr == i
            pruned = remove_small_objects(mask, 100)  # pruned[inter > 0] = 1
            final_labels_hr[mask > 0] = 0
            final_labels_hr[pruned > 0] = i

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_segmentation_image(data, "pre_final_labels", final_labels_hr - 1, img)

        final_labels_hr[final_labels_hr < 0] = 0
        final_labels = np.int32(final_labels_hr)

        self.transfer_labels(data, final_labels, len(final_masks), final_plane_parameters, final_rotations)

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_segmentation_image(data, "final_labels", np.int32(final_labels) - 1, img)
        
                
