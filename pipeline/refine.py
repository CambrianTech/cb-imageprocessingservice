from pipeline.core import PipelineStep

import cv2
import numpy as np
import os
from skimage.filters import threshold_sauvola
from scipy import ndimage, stats
from skimage.morphology import reconstruction
from cambrian import image_processing as ip
from skimage.transform import match_histograms
from skimage import filters
from skimage.morphology import watershed, disk
from skimage.color import rgb2gray, gray2rgb, label2rgb
from skimage.segmentation import mark_boundaries
from skimage import filters
from skimage.feature import canny
from skimage import measure
from skimage.segmentation import random_walker
from skimage.filters import threshold_multiotsu
from skimage.future import graph

from skimage.feature import peak_local_max

IM_LOGGING_ENABLED = False

def _log_image(name, image):
    if IM_LOGGING_ENABLED:
        cv2.imwrite(name, image)

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.
        
        The method expects that the mean color of `dst` is already computed.
        
        Parameters
        ----------
        graph : RAG
        The graph under consideration.
        src, dst : int
        The vertices in `graph` to be merged.
        n : int
        A neighbor of `src` or `dst` or both.
        
        Returns
        -------
        data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
        """
    
    diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}


def merge_mean_color(graph, src, dst):
    """Callback called before merging two nodes of a mean color distance graph.
        
        This method computes the mean color of `dst`.
        
        Parameters
        ----------
        graph : RAG
        The graph under consideration.
        src, dst : int
        The vertices in `graph` to be merged.
        """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])


class PipelineRefineResults(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "lighting", "kmeans_normals"]

    @property
    def output_keys(self) -> list:
        return ["lighting", "mask"]

    def run(self, data):
        shape = (1024, 1024)

        img = data["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = cv2.resize(img, shape)
        img_epf = cv2.edgePreservingFilter(img, flags=1, sigma_s=50, sigma_r=0.2)
#        cv2.imwrite('img_epf.png',img_epf)
        img_bw = cv2.cvtColor(img_epf, cv2.COLOR_BGR2GRAY)
        
        lighting_rgb = np.uint8(data["lighting"])
        lighting_rgb = cv2.resize(lighting_rgb, shape)
        lighting = lighting_rgb[:, :, 1]

        
        kmeans = data["kmeans_normals"]
        kmeans = cv2.resize(kmeans, shape)
#        cv2.imwrite('kmeans.png',kmeans)


        normals_up = kmeans[:,:,2]
        normals_up=cv2.threshold((normals_up).astype('uint8'),127,255,cv2.THRESH_TOZERO)[1]
#        cv2.imwrite('normals_up.png',normals_up)

        e1 = cv2.getTickCount()
        
        hed = data["hed"]
        hed = (255 * hed[0, 0]).astype("uint8")
        
        e2 = cv2.getTickCount()
        t = (e2 - e1)/cv2.getTickFrequency()
        print("hed took " + str(t) + " seconds")

#        cv2.imwrite("hed.png", hed)

        prob_mask_full = np.uint8(255*data["semantic_probs"][:, :, 0])
        # Sometimes there are border artifacts masks
        prob_mask_full[:, 510:512] = prob_mask_full[:, 508:510]
        prob_mask_full[510:512, :] = prob_mask_full[508:510, :]
        prob_mask_full = cv2.resize(prob_mask_full, shape)

#        cv2.imwrite('prob_mask_full.png',prob_mask_full)

#        edges = canny(img_bw, 3, 1, 25)
        edges = cv2.Canny(img_bw,100,200)
        edges = cv2.dilate(255*np.uint8(edges>0), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        _log_image('edges_o.png',255*np.uint8(edges>0))
        edges_hed = cv2.Canny(hed,100,200)
        edges_hed[:, 1020:1024] = edges_hed[:, 1015:1019]
        edges_hed[1020:1024, :] = edges_hed[1015:1019, :]
        edges_hed = cv2.dilate(255*np.uint8(edges_hed>0), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))

        _log_image('edges_hed.png',edges_hed)

        normals_up = filters.rank.median(normals_up, disk(5))
        edges_normals = cv2.Canny(normals_up,100,200)
        edges_normals = cv2.dilate(255*np.uint8(edges_normals>0), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        _log_image('edges_normals.png',255*np.uint8(edges_normals>0))

        edges = 255*np.uint8(edges>0)
        edges[edges_hed>0] = 255
        edges[edges_normals>0] = 255
        _log_image('edges.png',255*np.uint8(edges>0))
        
        thresholds = threshold_multiotsu(prob_mask_full, classes = 4)
        big_mask = 1-np.uint8(prob_mask_full>.01*255)
#        cv2.imwrite("big_mask_pre.png", 255*big_mask)
        big_mask = 255*ip.refine_mask_watershed(None, edges, big_mask, None, distance=0.05, max_value=1)

        # cv2.imwrite("big_mask.png", big_mask)

        isolated = np.uint8(prob_mask_full>thresholds[2])
        _log_image("isolated_pre.png", 255*isolated)

        isolated = 255*(ip.refine_mask_watershed(None, edges, isolated, None, distance=0.01, max_value=1))

        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(isolated, connectivity=8)
        sizes = stats[:, -1]
        sizes[0] = 0
        print(sizes)
        max_label = np.argmax(sizes)
        print(max_label)
        isolated = np.zeros(isolated.shape)

        for i in range(0, nb_components):
            if sizes[i] >= 64*64:
                  isolated[output == i] = 255

        isolated_small = cv2.erode(isolated, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (75,75)))
        isolated[edges>0] = 0
        markers = isolated
        markers[edges==0] = 255
#        cv2.imwrite("isolated_post.png", isolated_small)
#
        markers = cv2.erode(markers, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        markers[isolated_small>0] = 255

#        cv2.imwrite('markers.png',markers)

        #regions = np.digitize(hed, bins=thresholds)
        # cv2.imwrite('regions.png', 255*label2rgb(regions, image=img, kind='overlay'))
        trim_mask = np.zeros(prob_mask_full.shape)
        trim_mask[prob_mask_full> np.mean(prob_mask_full)] = prob_mask_full[prob_mask_full>np.mean(prob_mask_full)]
        trim_mask[big_mask>0] = 0
        trim_mask = np.uint8(trim_mask)
#        print(trim_mask)
#        cv2.imwrite('trim_mask.png', trim_mask)

        e1 = cv2.getTickCount()
        markers = ndimage.label(markers)
#        print(markers)
        markers = markers[0]
        ret = len(np.unique(markers))
#        ret, markers = cv2.connectedComponents(markers)
#        print(ret)
#        markers = markers + 1
#        cv2.imwrite('markers_l.png',255*np.uint8(markers))
        hed_rgb = cv2.cvtColor(hed, cv2.COLOR_GRAY2RGB)

#        labels = cv2.watershed(hed_rgb, markers)
        labels = watershed(hed, markers)

        watershed_mask = np.zeros(prob_mask_full.shape)

        for i in range(0, ret-1):
            a =  labels == i
            m = np.mean(trim_mask[a])
            watershed_mask[a] = m

        e2 = cv2.getTickCount()
        t = (e2 - e1)/cv2.getTickFrequency()
        print("watershed took " + str(t) + " seconds")

        _log_image('watershed.png', watershed_mask)


        watershed_mask = 255*np.uint8(watershed_mask > 127)
        watershed_mask[big_mask>0] = 0
#       ftriz cv2.imwrite("watershed_minusbig.png", watershed_mask)
        nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(watershed_mask, connectivity=8)
        sizes = stats[:, -1]
        sizes[0] = 0
        print(sizes)
        max_label = np.argmax(sizes)
        print(max_label)
        watershed_mask = np.zeros(isolated.shape)

        for i in range(0, nb_components):
            if sizes[i] >= 64*64:
                  watershed_mask[output == i] = 255

        # Fill small holes
        contours, hierarchy = cv2.findContours(watershed_mask.astype('uint8'), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        final_mask = np.zeros(shape)

        if len(contours) > 0:
            for contour in contours:
                area = cv2.contourArea(contour)
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    color = watershed_mask[cY, cX]/255
                    if area < 32*32 and watershed_mask[cY, cX] == 0:
                        color = 1

                    # compute the center of the contour

                    cv2.drawContours(final_mask, [contour], 0, color, -1, cv2.LINE_AA)

   
#                                mask = cv2.threshold(mask, t, 255, cv2.THRESH_BINARY)[1]
#        final_mask = cv2.GaussianBlur(final_mask, (3, 3), 0)
        final_mask = 255*(ip.refine_mask_watershed(None, img, final_mask, None, distance=0.02, max_value=1))
        final_mask = cv2.GaussianBlur(final_mask, (15, 15), 0)
        _, final_mask = cv2.threshold(final_mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#

        data["mask"] = final_mask

        _log_image('final_mask.png', data["mask"])

#        Remove hard edges from lighting

        smooth_lighting = ip.remove_grooves(lighting, data["mask"])

        blurred_mask = cv2.GaussianBlur(data["mask"], (31, 31), 15)
        blurred_lighting = cv2.GaussianBlur(smooth_lighting, (21, 21), 11)

        lighting = ip.alpha_blend(smooth_lighting, blurred_lighting, blurred_mask)
    
        data["lighting"] = lighting

        _log_image('lighting.png', lighting)
