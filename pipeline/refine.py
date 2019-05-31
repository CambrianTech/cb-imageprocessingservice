from pipeline.core import PipelineStep

import cv2
import numpy as np
from skimage.filters import threshold_sauvola
from scipy import ndimage, stats
from skimage.morphology import reconstruction
from cambrian import image_processing as ip


class PipelineRefineResults(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "lighting", "kmeans_normals"]

    @property
    def output_keys(self) -> list:
        return ["lighting", "semantic_probs", "mask"]

    def run(self, data):
        shape = (1024, 1024)

        img = data["image"]

        prob_mask_full = np.uint8(255*data["semantic_probs"][:, :, 0])

        lighting_rgb = np.uint8(data["lighting"])
        lighting = lighting_rgb[:, :, 1]

        smooth_lighting = ip.remove_grooves(lighting, prob_mask_full)

        blurred_mask = cv2.GaussianBlur(prob_mask_full, (31, 31), 15)
        blurred_lighting = cv2.GaussianBlur(smooth_lighting, (21, 21), 11)

        lighting = ip.alpha_blend(
            smooth_lighting, blurred_lighting, blurred_mask)

        data["lighting"] = lighting

        prob_mask = cv2.resize(prob_mask_full, shape)

        thresh, mask = cv2.threshold(
            prob_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        prob_mask = cv2.threshold(prob_mask, thresh, 255, 3)[1]
        img = cv2.resize(img, shape)

        thresh_s = threshold_sauvola(prob_mask, window_size=5, k=0.2)
        a = np.abs(thresh_s)
        a *= 1/np.amax(a)
        prob_mask = a

        distance = ndimage.distance_transform_edt(
            1.0-prob_mask/np.max(prob_mask))

        seed = np.copy(distance)/np.amax(distance)*prob_mask
        seed[1:-1, 1:-1] = (np.copy(distance) /
                            np.amax(distance)*prob_mask).min()
        mask = prob_mask

        dilated = reconstruction(seed, mask, method='dilation')
        mask = prob_mask-dilated

        # watershed some
        mask[mask > 0] = 1
        mask = ip.refine_mask_watershed(None, cv2.bilateralFilter(
            img, 21, 75, 75), mask, None, distance=0.02, max_value=1)

        mask = (255.*mask).astype('uint8')
        mask = cv2.GaussianBlur(mask, (15, 15), 0)
        t, mask = cv2.threshold(
            mask, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
        mask = cv2.threshold(mask, t, 255, cv2.THRESH_BINARY)[1]

        kmeans = data["kmeans_normals"]

        kmeans_gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(kmeans_gray, 127, 255, 0)
        nmask = np.zeros(mask.shape, np.uint8)

        # Get biggest normals contour
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            M = cv2.moments(cnt)
            if M["m00"] != 0:
                c_x = int(M["m10"] / M["m00"])
                c_y = int(M["m01"] / M["m00"])
                if kmeans[c_y, c_x][2] > 127:
                    nmask = ip.isolate_color(kmeans, kmeans[c_y, c_x])

        nmask = cv2.resize(nmask, shape)

        border = 50
        border_mask = cv2.copyMakeBorder(nmask, border, border,
                                         border, border, cv2.BORDER_REPLICATE)
        cv2.rectangle(border_mask, (51, 51), (
            border_mask.shape[1] - border - 1, border_mask.shape[0] - border - 1), 0, cv2.FILLED)

        mask = cv2.copyMakeBorder(
            mask, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
        mask = mask + border_mask

        img = cv2.copyMakeBorder(
            img, border, border, border, border, cv2.BORDER_CONSTANT, value=0)

        # Fill border holes
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) > 0:
            areas = [cv2.contourArea(c) for c in contours]
            max_index = np.argmax(areas)
            cnt = contours[max_index]
            # compute the center of the contour
            min_area = (shape[0] * shape[1]) / 200

            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    # compute the center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        color = 0 if mask[cY, cX] > 0 else 255
                        mask = cv2.fillPoly(mask, pts=[contour], color=color)

        mask = ip.crop_image(mask, border)
        img = ip.crop_image(img, border)

        # Fill small holes
        contours, hierarchy = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if len(contours) > 0:
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < min_area:
                    # compute the center of the contour
                    M = cv2.moments(contour)
                    if M["m00"] != 0:
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                        color = 0 if mask[cY, cX] > 0 else 255
                        mask = cv2.fillPoly(mask, pts=[contour], color=color)

        mask = cv2.GaussianBlur(mask, (5, 5), 0)

        data["mask"] = mask
