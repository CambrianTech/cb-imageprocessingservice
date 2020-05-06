from pipeline.core import PipelineStep
import cv2
import numpy as np
from scipy import ndimage
from cambrian import image_processing as ip
from skimage.morphology import watershed, disk
from skimage import filters
from skimage.filters import threshold_multiotsu
import os

IM_LOGGING_ENABLED = True


def _log_image(name, image):
    if IM_LOGGING_ENABLED:
        cv2.imwrite(name, image)


class PipelineRefinePlaneMasks(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "hed"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):

        hed = data["hed"]

        w, h = hed.shape
        shape = (h,w)

        plane_masks = data["planes"]["masks"][:, 80:560]

        number_planes = len(plane_masks)
        resized_masks = np.zeros((number_planes, w, h))
        plane_shape = plane_masks[0].shape

        for i in range(number_planes):
            resized_masks[i] = cv2.resize(plane_masks[i], shape)

        edges_hed = cv2.Canny(hed, 0, 254)

        edges_hed = cv2.dilate(255 * np.uint8(edges_hed > 0),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)))

        markers = np.uint8(edges_hed == 0)

        markers = filters.rank.median(markers, disk(2))

        markers = ndimage.label(markers)
        markers = markers[0]
        labels = watershed(hed, markers)

        ulabels = np.unique(labels)
        nlabels = ulabels[-1]

        detection_features = []
        clean_masks = np.uint8(np.zeros_like(resized_masks))

        for d in range(number_planes):
            detection_features.append(ndimage.mean(resized_masks[d], labels=labels, index=ulabels))

        for label in range(1, nlabels + 1):
            label_mask_i = 0
            label_mask_max = 0
            label_mask = label == labels

            for d in range(number_planes):
                m = detection_features[d][label - 1]

                if m > .5:
                    label_mask_i = d
                    label_mask_max = m
                    continue

                if m > label_mask_max:
                    label_mask_i = d
                    label_mask_max = m

            if label_mask_max == 0:
                for d in range(number_planes):
                    dia_label = cv2.blur(255 * np.uint8(label_mask), (5, 5))
                    d_mask = resized_masks[d][dia_label > 0]
                    m = np.mean(d_mask)

                    if m > .5:
                        label_mask_i = d
                        label_mask_max = m
                        continue

                    if m > label_mask_max:
                        label_mask_i = d
                        label_mask_max = m

            if label_mask_max > 0.4:
                clean_masks[label_mask_i][label_mask] = 1

            # print("label", label, label_mask_i, label_mask_max)

            for d in range(number_planes):
                data["planes"]["masks"][:, 80:560][d] = cv2.resize(clean_masks[d], (plane_shape[1], plane_shape[0]))
        #
        # for d in range(number_planes):
        #     mask = resized_masks[d]
        #     blank = np.zeros_like(mask)
        #
        #     for label in range(1, nlabels + 1):
        #         label_mask = label == labels
        #         mean = np.mean(mask[label_mask])
        #         if(mean > .5):
        #             blank[label_mask] = 1

            # plane_masks[d] = cv2.resize(np.uint8(blank), (plane_shape[1], plane_shape[0]))
            # _log_image(os.path.join(data["results_local_dir"],'masks'+str(d)+'.png'), (255*plane_masks[d]))

        # data["planes"]["masks"][:, 80:560] = plane_masks




