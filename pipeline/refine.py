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
        cv2.imwrite('logging/' + name, image)



class PipelineRefineResults(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "hed", "normals"]

    @property
    def output_keys(self) -> list:
        return ["mask", "lighting"]

    def run(self, data):

        hed = data["hed"]
        _log_image('hed.png', hed)
        normals = np.uint8(data["normals"])
        _log_image('normals.png', normals)
        w, h = hed.shape
        shape = (h,w)

        img = data["image"]
        img = cv2.resize(img, shape)
        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        semantic_mask = data["semantic_probs"]

        prob_mask_full = np.uint8(255.*(semantic_mask[3]+semantic_mask[28]))

        # Sometimes there are border artifacts masks
        # prob_mask_full[:, 510:512] = prob_mask_full[:, 508:510]
        # prob_mask_full[510:512, :] = prob_mask_full[508:510, :]
        prob_mask_full = cv2.resize(prob_mask_full, shape)

        _log_image('prob_mask_full.png', prob_mask_full)

        edges = cv2.Canny(img_bw, 100, 200)

        _log_image('edges_o.png', 255*np.uint8(edges > 0))
        edges_hed = cv2.Canny(hed, 10, 250)

        _log_image('edges_hed.png', edges_hed)

        edges = 255*np.uint8(edges > 0)
        edges[edges_hed > 0] = 255
        edges = cv2.dilate(255*np.uint8(edges > 0),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        _log_image('edges.png', 255*np.uint8(edges > 0))

        thresholds = threshold_multiotsu(prob_mask_full, classes=4)
        big_mask = 1-np.uint8(prob_mask_full > .01*255)
        big_mask = 255 * \
            ip.refine_mask_watershed(
                None, edges, big_mask, None, distance=0.05, max_value=1)
        _log_image('big_mask.png', big_mask)

        isolated = np.uint8(prob_mask_full > thresholds[2])
        _log_image("isolated_pre.png", 255*isolated)

        isolated = 255*(ip.refine_mask_watershed(None, edges,
                                                 isolated, None, distance=0.01, max_value=1))

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            isolated, connectivity=8)
        sizes = stats[:, -1]
        sizes[0] = 0

        isolated = np.zeros(isolated.shape)

        for i in range(0, nb_components):
            if sizes[i] >= 64*64:
                isolated[output == i] = 255

        isolated_small = cv2.erode(
            isolated, cv2.getStructuringElement(cv2.MORPH_RECT, (75, 75)))

        isolated[edges > 0] = 0
        markers = isolated
        markers[edges == 0] = 255

        markers = cv2.erode(markers, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)))
        markers[isolated_small > 0] = 255

        trim_mask = np.zeros(prob_mask_full.shape)
        trim_mask[prob_mask_full > np.mean(
            prob_mask_full)] = prob_mask_full[prob_mask_full > np.mean(prob_mask_full)]
        trim_mask[big_mask > 0] = 0
        trim_mask = np.uint8(trim_mask)

        markers = ndimage.label(markers)

        markers = markers[0]

        labels = watershed(hed, markers)

        watershed_mask = np.zeros(prob_mask_full.shape)

        avgs = ndimage.mean(trim_mask, labels=labels, index=np.unique(labels))
        mass = ndimage.sum(trim_mask, labels=labels, index=np.unique(labels))
        max_index = np.argmax(mass)

        for i in range(0, len(np.unique(labels))):
            watershed_mask[labels == i+1] = avgs[i]

        _log_image('watershed_pre.png', watershed_mask)
        watershed_mask = 255*np.uint8(watershed_mask > max(min(127,avgs[max_index]-5),45))
        _log_image('watershed.png', watershed_mask)
        watershed_mask[big_mask > 0] = 0

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            watershed_mask, connectivity=8)
        sizes = stats[:, -1]
        sizes[0] = 0

        watershed_mask = np.zeros(isolated.shape)

        for i in range(0, nb_components):
            if sizes[i] >= 64*64:
                watershed_mask[output == i] = 255

        # Fill small holes
        contours, hierarchy = cv2.findContours(watershed_mask.astype(
            'uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros(isolated.shape)

        if len(contours) > 0:
            for contour in contours:
                area = cv2.contourArea(contour)
                test_mask = cv2.drawContours(
                    final_mask, [contour], 0, 1, -1, cv2.LINE_AA)
                nz = cv2.countNonZero(watershed_mask[test_mask > 0])/255
                if nz > 9 or area < 32*32:
                    final_mask[test_mask > 0] = 1

        final_mask = 255*(ip.refine_mask_watershed(None, img,
                                                   final_mask, None, distance=0.02, max_value=1))
        final_mask = cv2.GaussianBlur(final_mask, (15, 15), 0)

        _, final_mask = cv2.threshold(
            final_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        data["mask"] = final_mask

        _log_image('final_mask.png', data["mask"])

        lighting_rgb = np.uint8(data["lighting"])
        lighting_rgb = cv2.resize(lighting_rgb, shape)
        lighting = lighting_rgb[:, :, 1]

        # Remove hard edges from lighting
        smooth_lighting = ip.remove_grooves(lighting, data["mask"])

        blurred_mask = cv2.GaussianBlur(data["mask"], (31, 31), 15)

        blurred_lighting = cv2.GaussianBlur(smooth_lighting, (21, 21), 11)

        lighting = ip.alpha_blend(
            smooth_lighting, blurred_lighting, blurred_mask)

        data["lighting"] = lighting

        _log_image('lighting.png', lighting)
