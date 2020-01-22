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


class PipelineRefineResults(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "lighting", "kmeans_normals", "hed"]

    @property
    def output_keys(self) -> list:
        return ["lighting", "mask"]

    def run(self, data):
        shape = (1024, 1024)

        img = data["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, shape)

        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        lighting_rgb = np.uint8(data["lighting"])
        lighting_rgb = cv2.resize(lighting_rgb, shape)
        lighting = lighting_rgb[:, :, 1]

        kmeans = data["kmeans_normals"]
        kmeans = cv2.resize(kmeans, shape)

        normals_up = kmeans[:, :, 2]
        normals_up = cv2.threshold((normals_up).astype(
            'uint8'), 127, 255, cv2.THRESH_TOZERO)[1]
        _log_image('normals_up.png', normals_up)

        hed = data["hed"]
        _log_image('hed.png', hed)

        prob_mask_full = np.uint8(255*data["semantic_probs"][:, :, 0])

        # Sometimes there are border artifacts masks
        prob_mask_full[:, 510:512] = prob_mask_full[:, 508:510]
        prob_mask_full[510:512, :] = prob_mask_full[508:510, :]
        prob_mask_full = cv2.resize(prob_mask_full, shape)

        _log_image('prob_mask_full.png', prob_mask_full)

        edges = cv2.Canny(img_bw, 100, 200)
        edges = cv2.dilate(255*np.uint8(edges > 0),
                           cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        _log_image('edges_o.png', 255*np.uint8(edges > 0))
        edges_hed = cv2.Canny(hed, 100, 200)
        edges_hed[:, 1020:1024] = edges_hed[:, 1015:1019]
        edges_hed[1020:1024, :] = edges_hed[1015:1019, :]
        edges_hed = cv2.dilate(255*np.uint8(edges_hed > 0),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

        _log_image('edges_hed.png', edges_hed)

        normals_up = filters.rank.median(normals_up, disk(5))

        edges_normals = cv2.Canny(normals_up, 100, 200)
        edges_normals = cv2.dilate(
            255*np.uint8(edges_normals > 0), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        _log_image('edges_normals.png', 255*np.uint8(edges_normals > 0))

        edges = 255*np.uint8(edges > 0)
        edges[edges_hed > 0] = 255
        edges[edges_normals > 0] = 255
        _log_image('edges.png', 255*np.uint8(edges > 0))

        thresholds = threshold_multiotsu(prob_mask_full, classes=4)
        big_mask = 1-np.uint8(prob_mask_full > .01*255)
        big_mask = 255 * \
            ip.refine_mask_watershed(
                None, edges, big_mask, None, distance=0.05, max_value=1)

        isolated = np.uint8(prob_mask_full > thresholds[2])
        _log_image("isolated_pre.png", 255*isolated)

        isolated = 255*(ip.refine_mask_watershed(None, edges,
                                                 isolated, None, distance=0.01, max_value=1))

        nb_components, isolated_cc, stats, centroids = cv2.connectedComponentsWithStats(
            isolated, connectivity=8)
        sizes = stats[:, -1]

        isolated = np.zeros(isolated.shape)

        for i in range(1, nb_components):
            if sizes[i] > 64*64:
                isolated[isolated_cc == i] = 255

        _log_image("isolated.png", isolated)
        isolated_small = cv2.erode(
            isolated, cv2.getStructuringElement(cv2.MORPH_RECT, (75, 75)))

        _log_image("isolated_small.png", isolated_small)

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

        watershed_mask = np.uint8(isolated_cc == 1)

        avgs = ndimage.mean(trim_mask, labels=labels, index=np.unique(labels))

        for i in range(0, len(np.unique(labels))):

            if avgs[i] > 127:
                watershed_mask[labels == i+1] = 1
                continue

            if cv2.countNonZero(isolated_small[labels == i+1]) > 64:
                watershed_mask[labels == i+1] = 1

        _log_image('watershed.png', 255*watershed_mask)

        watershed_mask[big_mask > 0] = 0

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(
            watershed_mask, connectivity=8)
        sizes = stats[:, -1]
        sizes[0] = 0

        watershed_mask = np.zeros(watershed_mask.shape)

        for i in range(0, nb_components):
            if sizes[i] >= 128*128:
                watershed_mask[output == i] = 255
        _log_image('final_watershed.png', watershed_mask)

        # Fill small holes
        contours, hierarchy = cv2.findContours(watershed_mask.astype(
            'uint8'), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        final_mask = np.zeros(shape)
        if len(contours) > 0:
            for contour in contours:
                area = cv2.contourArea(contour)
                test_mask = cv2.drawContours(
                    final_mask, [contour], 0, 1, -1, cv2.LINE_AA)
                nz = cv2.countNonZero(watershed_mask[test_mask > 0]) / 255
                if nz > 9 or area < 32 * 32:
                    final_mask[test_mask > 0] = 1

        final_mask = 255*(ip.refine_mask_watershed(None, img,
                                                   final_mask, None, distance=0.02, max_value=1))
        final_mask = cv2.GaussianBlur(final_mask, (15, 15), 0)

        _, final_mask = cv2.threshold(
            final_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        data["mask"] = final_mask

        img[final_mask > 0] = (0, 255, 255)
        _log_image('img.png', img)
        _log_image('final_mask.png', data["mask"])

        # Remove hard edges from lighting
        smooth_lighting = ip.remove_grooves(lighting, data["mask"])

        blurred_mask = cv2.GaussianBlur(data["mask"], (31, 31), 15)
        blurred_lighting = cv2.GaussianBlur(smooth_lighting, (21, 21), 11)

        lighting = ip.alpha_blend(
            smooth_lighting, blurred_lighting, blurred_mask)

        data["lighting"] = lighting

        _log_image('lighting.png', lighting)
