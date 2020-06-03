from pipeline.core import PipelineStep
import cv2
import numpy as np
from scipy import ndimage
from cambrian import image_processing as ip
from skimage.morphology import watershed, disk
from skimage import filters, img_as_float
from skimage.filters import threshold_multiotsu, frangi
import pickle
import os

IM_LOGGING_ENABLED = False

def _log_image(name, image):
    if IM_LOGGING_ENABLED:
        cv2.imwrite('results/'+name, image)

class PipelineRefinePlaneMasks(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "hed", "mask"]

    @property
    def output_keys(self) -> list:
        return []


    def run(self, data):

        hed = data["hed"]
        prob_mask_full = np.uint8(255*data["semantic_probs"][:, :, 0])
        plane_masks = data["planes"]["masks"]
        img = data["image"]
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        number_planes = len(plane_masks)
        print("number of planes: ", number_planes)

        w, h = hed.shape
        shape = (h, w)

        # Sometimes there are border artifacts masks
        prob_mask_full[:, -5:prob_mask_full.shape[1]] = prob_mask_full[:, -11:-6]
        prob_mask_full[-5:prob_mask_full.shape[0], :] = prob_mask_full[-11:-6, :]

        # Chairleg detection using frangi filter
        # t = time()
        frangi_mask = img_as_float(prob_mask_full)
        frangi_black = np.uint8(
            255 * frangi(255. * frangi_mask, sigmas=(1, 3), scale_step=.5, beta=0.3, gamma=15, black_ridges=True))
        frangi_white = np.uint8(
            255 * frangi(255. * frangi_mask, sigmas=(1, 3), scale_step=.5, beta=0.3, gamma=15, black_ridges=False))
        # print("Frangi filter took %.2f seconds" % (time() - t))

        prob_mask_full = cv2.add(prob_mask_full, frangi_white)
        prob_mask_full = cv2.subtract(prob_mask_full, frangi_black)
        frangi_white = cv2.resize(frangi_white, shape)
        frangi_black = cv2.resize(frangi_black, shape)
        _log_image('frangi_white.png', frangi_white)
        _log_image('frangi_black.png', frangi_black)

        prob_mask_full = cv2.resize(prob_mask_full, shape)

        _log_image('prob_mask_frangi.png', prob_mask_full)

        img_bw_smooth = filters.rank.median(img_bw, disk(2))
        edges = cv2.Canny(img_bw_smooth, 100, 200)
        edges = cv2.resize(edges, shape)

        _log_image('edges_img.png', 255 * np.uint8(edges > 0))
        edges = cv2.dilate(255 * np.uint8(edges > 0),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

        edges_hed = cv2.Canny(hed, 10, 250)
        edges_hed[:, -5:-1] = edges_hed[:, -10:-6]
        edges_hed[-5:-1, :] = edges_hed[-10:-6, :]
        frangi_white[edges_hed > 0] = 0
        frangi_black[edges_hed > 0] = 0

        edges_hed = cv2.dilate(255 * np.uint8(edges_hed > 0),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))

        _log_image('edges_hed.png', edges_hed)

        resized_masks = np.zeros((number_planes, shape[1], shape[0]))
        print(resized_masks.shape)
        core_masks = np.zeros_like(resized_masks)
        core_edges = np.zeros((shape[1], shape[0]))

        floor_mask_index = -1
        floor_intersection = 0
        otsu_thresholds = []
        for d in range(number_planes):
            resized_masks[d] = cv2.resize(np.uint8(255.*plane_masks[d]), shape)
            _log_image('resized_masks_' + str(d) + '.png', resized_masks[d])
            thresholds = threshold_multiotsu(np.uint8(resized_masks[d]), classes=4)
            otsu_thresholds.append(thresholds)
            t = np.copy(resized_masks[d])
            resized_masks[d][t > thresholds[-1]] = 255
            core_masks[d][t > thresholds[-1]] = 255
            inter = cv2.countNonZero(core_masks[d][prob_mask_full > 127])
            if inter > floor_intersection:
                floor_mask_index = d
                floor_intersection = inter
            resized_masks[d][t < thresholds[0]] = 0

            contours, hierarchy = cv2.findContours(np.uint8(core_masks[d]), cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(core_edges, contours, -1, 255, 2)
            core_masks[d][edges_hed > 0] = 0

            _log_image('core_edges.png', core_edges)

        print("floor mask index: ", floor_mask_index)

        edges = 255 * np.uint8(edges > 0)
        edges[edges_hed > 0] = 255

        _log_image('edges.png', 255 * np.uint8(edges > 0))

        markers = 255 * np.uint8(edges_hed == 0)

        for d in range(number_planes):
            markers[core_masks[d] > 0] = 255
        markers[frangi_black > 127] = 255
        markers[core_edges > 0] = 0

        markers = filters.rank.median(markers, disk(1))

        markers = ndimage.label(markers)

        markers = markers[0]

        _log_image('markers.png', markers)

        labels = watershed(hed, markers)

        watershed_mask = np.zeros(prob_mask_full.shape)

        ulabels = np.unique(labels)
        nlabels = ulabels[-1]

        detection_features = []
        resized_masks[floor_mask_index] = prob_mask_full
        for d in range(number_planes):
            detection_features.append(ndimage.mean(
                resized_masks[d], labels=labels, index=ulabels))

        clean_masks = np.zeros_like(resized_masks)
        for label in range(1, nlabels + 1):
            label_mask_i = 0
            label_mask_max = 0
            label_mask = label == labels

            for d in range(number_planes):
                m = detection_features[d][label - 1] / 255.
                if m > .5:
                    label_mask_i = d
                    label_mask_max = m
                    continue

                if m > label_mask_max:
                    label_mask_i = d
                    label_mask_max = m

            if label_mask_max == 0:
                for d in range(number_planes):
                    dia_label = cv2.blur(255 * np.uint8(label_mask), (15, 15))
                    d_mask = resized_masks[d][dia_label > 0]
                    m = np.mean(d_mask)

                    if m > .5:
                        label_mask_i = d
                        label_mask_max = m
                        continue

                    if m > label_mask_max:
                        label_mask_i = d
                        label_mask_max = m

            clean_masks[label_mask_i][label_mask] = label_mask_max
            # print("label", label, label_mask_i, label_mask_max)
        resized_masks = np.zeros((number_planes, img.shape[0], img.shape[1]))
        for d in range(number_planes):
            _log_image('clean_masks_' + str(d) + '.png', np.uint8(255. * clean_masks[d]))
            t = otsu_thresholds[d]

            b = cv2.resize(np.uint8(clean_masks[d] > (t[1] - 5) / 255.), (img.shape[1], img.shape[0]))
            b = cv2.GaussianBlur(b, (15, 15), 0)

            _, resized_masks[d] = cv2.threshold(
                b, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        data["planes"]["masks"] = np.zeros_like(resized_masks)
        for d in range(number_planes):
            data["planes"]["masks"][d] = np.uint8(resized_masks[d])

            rect = cv2.boundingRect(np.uint8(data["planes"]["masks"][d]))
            data["planes"]["detection"][d, 0:4] = [
                rect[1], rect[0], rect[1] + rect[3], rect[0] + rect[2]
            ]
