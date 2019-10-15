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


class CropLayer(object):
    def __init__(self, params, blobs):
        # initialize our starting and ending (x, y)-coordinates of
        # the crop
        self.startX = 0
        self.startY = 0
        self.endX = 0
        self.endY = 0
    
    def getMemoryShapes(self, inputs):
        # the crop layer will receive two inputs -- we need to crop
        # the first input blob to match the shape of the second one,
        # keeping the batch size and number of channels
        (inputShape, targetShape) = (inputs[0], inputs[1])
        (batchSize, numChannels) = (inputShape[0], inputShape[1])
        (H, W) = (targetShape[2], targetShape[3])
        
        # compute the starting and ending crop coordinates
        self.startX = int((inputShape[3] - targetShape[3]) / 2)
        self.startY = int((inputShape[2] - targetShape[2]) / 2)
        self.endX = self.startX + W
        self.endY = self.startY + H
        
        # return the shape of the volume (we'll perform the actual
        # crop during the forward pass
        return [[batchSize, numChannels, H, W]]
    
    def forward(self, inputs):
        # use the derived (x, y)-coordinates to perform the crop
        return [inputs[0][:, :, self.startY:self.endY,
                          self.startX:self.endX]]



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
        protoPath = os.path.sep.join(["hed_model", "deploy.prototxt"])
        modelPath = os.path.sep.join(["hed_model", "hed_pretrained_bsds.caffemodel"])
        net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)
        cv2.dnn_registerLayer("Crop", CropLayer)


        blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(1024,1024),
                                     mean=(104.00698793, 116.66876762, 122.67891434),
                                     swapRB=False, crop=True)
        net.setInput(blob)
        hed = net.forward()
        
        cv2.dnn_unregisterLayer("Crop")
        hed_32 = np.int32(255 * hed[0, 0])
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






#        lighting_bw = lighting_rgb[:,:,2]
#
#        cv2.imwrite('lighting_rgb.png', lighting_rgb)
#        cv2.imwrite('lighting_bw.png', lighting_bw)



        

#        prob_pm = np.uint8(255*np.amax(data["semantic_probs"],axis=-1))
#        prob_pm = cv2.edgePreservingFilter(prob_pm, flags=1, sigma_s=5, sigma_r=0.1)
#        prob_pm[prob_pm> .95*255] = 0
##        prob_pm = prob_mask_full
#        cv2.imwrite('prob_pm.png', prob_pm)
#        cv2.imwrite('prob_mask_full.png', prob_mask_full)
##        prob_mask = cv2.edgePreservingFilter(prob_pm, flags=1, sigma_s=5, sigma_r=0.1)
#        prob_mask = prob_pm
#        thresholds = threshold_multiotsu(prob_mask, classes = 3)
#        print("sthresholds")
#        print(np.mean(prob_pm))
#        print(thresholds)
#        regions = np.digitize(prob_mask, bins=thresholds)
#        print(regions)
#        print(np.amax(regions))
#        print(img_bw.shape)
#        cv2.imwrite('regions.png', 255.0*label2rgb(regions, image=prob_pm, kind='overlay'))
#
##        thresh, mask = cv2.threshold((prob_mask).astype('uint8'), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##        prob_mask[mask<255] = 0
#        cv2.imwrite('prob_epf.png', prob_mask)
#
#        z = prob_mask.reshape((-1, 1))
#        k = 100
#        # convert to np.float32
#        z = np.float32(z)
#
#        # define criteria, number of clusters(K) and apply kmeans()
#        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
#        compactness, labels, centers = cv2.kmeans(z, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS)
#        # Now convert back into uint8, and make original image
#        centers = np.uint8(centers)
#        res = centers[labels.flatten()]
#        res = res.reshape((prob_mask.shape))
#
#        distance = ndimage.distance_transform_edt(prob_mask)
#        cv2.imwrite("d.png", distance*prob_mask/np.amax(distance))
#        d = 1.0*distance*prob_mask/np.amax(distance)
#        a = label2rgb(res, image=prob_mask.astype('uint8'), kind='overlay')
#        prob_mask = label2rgb(res, image=d.astype('uint8'), kind='avg')
##        print(np.amax(prob_mask))
#        cv2.imwrite("strat_labeled.png", prob_mask.astype('uint8'))
#        cv2.imwrite("s_labeled.png", (255.0*a/np.amax(a)).astype('uint8'))
##        markers = cv2.GaussianBlur(255*np.uint8(markers>0), (15, 15), 0)
#        thresh, markers = cv2.threshold(prob_mask, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)
#        cv2.imwrite("markers.png", markers)
##        return
##        thresh, mask = cv2.threshold((255.0*prob_mask).astype('uint8'), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##        prob_mask = cv2.threshold((255.0*prob_mask).astype('uint8'), thresh, 255, 3)[1]
#
##        cv2.imwrite('thresh_s.png', prob_mask)
#
#        thresh_s = threshold_sauvola(prob_mask, window_size=5, k=0.2)
#        a = np.abs(thresh_s)
#        a *= 1/np.amax(a)
#        prob_mask = a
#        thresh, mask = cv2.threshold((255.0*prob_mask).astype('uint8'), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#        prob_mask = cv2.threshold((255.0*prob_mask).astype('uint8'), thresh, 255, 3)[1]
#
#        cv2.imwrite('thresh_s.png', prob_mask)
#
#
#        distance = ndimage.distance_transform_edt(np.uint8((prob_mask - np.amin(prob_mask[prob_mask>0]))))
#        cv2.imwrite('distance.png', distance)
#        distance = distance/np.amax(distance)
#        seed = distance*prob_mask
#        print(seed.min())
#        seed[1:-1, 1:-1] = seed.min()
#
#        cv2.imwrite('seed.png', 255*np.uint8(seed>0))
#
#        mask = prob_mask
#
#        dilated = reconstruction(seed, mask, method='dilation').astype('uint8')
##        print(dilated)
#        cv2.imwrite('dilated.png', dilated)
#        mask = mask - dilated

#        dilated = filters.rank.gradient(dilated.astype('uint8'), disk(5))
        #        gradient= cv2.edgePreservingFilter(gradient, flags=1, sigma_s=9, sigma_r=0.1)
#        markers =ndimage.label(dilated<50)[0]

#
#        labels = watershed(dilated, markers)
#        cv2.imwrite("s_labeled.png", 255*label2rgb(labels, image=dilated, kind='overlay'))
#        mask= filters.rank.median(np.uint8(255*mask/np.amax(mask)), disk(3))
#        mask = cv2.threshold(dilated, np.amin(dilated[dilated>0])+45, 255, cv2.THRESH_BINARY)[1]
#        cv2.imwrite('pre_water_mask.png', mask)
#        mask = ndimage.distance_transform_edt(mask)
#        not_mask = ndimage.distance_transform_edt(np.uint8(mask==0))
#        cv2.imwrite('distancen.png', 255*np.uint8(not_mask))
#
#        # watershed some
#        pre_markers = mask > 0
#        cv2.imwrite('distance2.png', 255*np.uint8(pre_markers))
#        pre_markers[not_mask>100] = True
#        cv2.imwrite('distance3.png', 255*np.uint8(pre_markers))
#        markers =ndimage.label(pre_markers)[0]
#        cv2.imwrite('pre_water_mask_bin.png', 255*np.uint8(markers))

#        smooth_img = cv2.edgePreservingFilter(img, flags=1, sigma_s=25, sigma_r=0.2)
#        dimg = filters.rank.median(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), disk(3))
#        gradient = filters.rank.gradient(dimg, disk(5))
##        gradient= cv2.edgePreservingFilter(gradient, flags=1, sigma_s=9, sigma_r=0.1)
#        markers =ndimage.label(gradient<5)[0]
#
#
#        labels = watershed(prob_mask_full, markers)
#        cv2.imwrite("w_labeled.png", 255*label2rgb(labels, image=img, kind='overlay'))
#        print(labels)
#        mask = (labels != 0)
#        mask = cv2.watershed(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype('float32'), markers)
#        mask = ip.refine_mask_watershed(None, , mask, None, distance=0.02, max_value=1)
#        mask = ip.refine_mask_watershed(None, cv2.bilateralFilter(
#            img, 21, 75, 75), mask, None, distance=0.02, max_value=1)

#        mask = 255*(labels != 0).astype('uint8')
#        mask = cv2.GaussianBlur(mask, (15, 15), 0)
#        t, mask = cv2.threshold(
#            mask, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

        
#        cv2.imwrite('post_water_mask_bin.png', mask)
#
#        kmeans = data["kmeans_normals"]
#
#        kmeans_gray = cv2.cvtColor(kmeans, cv2.COLOR_BGR2GRAY)
#        _, thresh = cv2.threshold(kmeans_gray, 127, 255, 0)
#        nmask = np.zeros(mask.shape, np.uint8)
#
#        # Get biggest normals contour
#        contours, _ = cv2.findContours(
#            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        if len(contours) > 0:
#            areas = [cv2.contourArea(c) for c in contours]
#            max_index = np.argmax(areas)
#            cnt = contours[max_index]
#            M = cv2.moments(cnt)
#            if M["m00"] != 0:
#                c_x = int(M["m10"] / M["m00"])
#                c_y = int(M["m01"] / M["m00"])
#                if kmeans[c_y, c_x][2] > 127:
#                    nmask = ip.isolate_color(kmeans, kmeans[c_y, c_x])
#
#        nmask = cv2.resize(nmask, shape)
#
#        border = 50
#        border_mask = cv2.copyMakeBorder(nmask, border, border,
#                                         border, border, cv2.BORDER_REPLICATE)
#        cv2.rectangle(border_mask, (51, 51), (
#            border_mask.shape[1] - border - 1, border_mask.shape[0] - border - 1), 0, cv2.FILLED)
#
#        mask = cv2.copyMakeBorder(
#            mask, border, border, border, border, cv2.BORDER_CONSTANT, value=0)
#        mask = mask + border_mask
#
#        # Fill border holes
#        contours, hierarchy = cv2.findContours(
#            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#        if len(contours) > 0:
#            areas = [cv2.contourArea(c) for c in contours]
#            max_index = np.argmax(areas)
#            cnt = contours[max_index]
#            # compute the center of the contour
#            min_area = (shape[0] * shape[1]) / 200
#
#            for contour in contours:
#                area = cv2.contourArea(contour)
#                if area < min_area:
#                    # compute the center of the contour
#                    M = cv2.moments(contour)
#                    if M["m00"] != 0:
#                        cX = int(M["m10"] / M["m00"])
#                        cY = int(M["m01"] / M["m00"])
#                        color = 0 if mask[cY, cX] > 0 else 255
#                        mask = cv2.fillPoly(mask, pts=[contour], color=color)
#
#        mask = ip.crop_image(mask, border)
#
#        # Fill small holes
#        contours, hierarchy = cv2.findContours(
#            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#
#        if len(contours) > 0:
#            for contour in contours:
#                area = cv2.contourArea(contour)
#                if area < min_area:
#                    # compute the center of the contour
#                    M = cv2.moments(contour)
#                    if M["m00"] != 0:
#                        cX = int(M["m10"] / M["m00"])
#                        cY = int(M["m01"] / M["m00"])
#                        color = 0 if mask[cY, cX] > 0 else 255
#                        mask = cv2.fillPoly(mask, pts=[contour], color=color)
#
#        mask = cv2.GaussianBlur(mask, (5, 5), 0)
#
#        data["mask"] = mask
##
#
##        Remove hard edges from lighting
#        lighting_rgb = np.uint8(data["lighting"])
#        lighting_rgb = cv2.resize(lighting_rgb, shape)
#        lighting_bw = lighting_rgb[:,:,2]
#
#        cv2.imwrite('lighting_rgb.png', lighting_rgb)
#        cv2.imwrite('lighting_bw.png', lighting_bw)
#
#        img_v = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,2]
#
#        cv2.imwrite('img_v.png', img_v)
#
##        lighting_bw = img_v
##        img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
##        lighting_bw = filters.rank.median(lighting_bw, disk(3))
##        cv2.imwrite("denoised.png", lighting_bw)
#
#        edges_lighting = canny(lighting_bw, 3, 1, 25)
#
##        print("gradient")
##        print(gradient.dtype)
##        print(gradient)
##        pre_markers = gradient > 10
##        print(pre_markers.dtype)
##        cv2.imwrite('markers2.png', 255*np.uint8(pre_markers))
##        markers =  ndimage.label(pre_markers)[0]
##        print(markers)
##        cv2.imwrite('pre_water_mask_bin2.png', markers)
##        gradient = cv2.adaptiveThreshold(gradient,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,35,-1)
##        gradient = filters.rank.median(gradient, disk(5))
##        cv2.imwrite("gradient.png", gradient)
#        edges_lighting = 255*(edges_lighting>0).astype('uint8')
#        edges_lighting = cv2.GaussianBlur(edges_lighting, (5, 5), 0)
#        _, edges_lighting= cv2.threshold(edges_lighting, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
#
#        cv2.imwrite("edges_lighting.png", edges_lighting)
#
#        lighting_inpaint = cv2.inpaint(cv2.GaussianBlur(lighting_bw, (11, 11), 0),edges_lighting ,3,cv2.INPAINT_NS)
##        lighting_inpaint = cv2.GaussianBlur(lighting_inpaint, (5, 5), 0)
#        lighting_epf = cv2.edgePreservingFilter(lighting_inpaint, flags=1, sigma_s=50, sigma_r=0.4)
#
#        cv2.imwrite("lighting_epf.png", lighting_epf)
##        lighting_inpaint= cv2.edgePreservingFilter(lighting_inpaint, flags=1, sigma_s=15, sigma_r=0.2)
#
#        cv2.imwrite("lighting_inpaint.png", lighting_inpaint)
#        lighting = lighting_epf

       
       
       
       
#                img_epf =cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#        pre_markers = cv2.adaptiveThreshold(hed,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,25,0)
#        pre_markers[edges>0] = 0
#
#        markers = ndimage.label(pre_markers)[0]
#        cv2.imwrite("markers.png", markers)
#        labels = watershed(edges, markers)
#
#        cv2.imwrite("watershed_mask.png", 255*mark_boundaries(img,labels))
#
#
##        cv2.imwrite("canny.png", edges)
##        thresh, _ = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
##        edges_s = 255*np.uint8(gradient > thresh-10)
##        cv2.imwrite('edges_s.png', edges_s)
#        inpaint_mask = (255-pre_markers).astype('uint8')
##        inpaint_mask[edges>0] = 255
#        img_v = cv2.inpaint(lighting_rgb[:, :, 1].astype('uint8'),inpaint_mask ,3,cv2.INPAINT_NS)
#        prob_mask_full = cv2.inpaint(prob_mask_full,cv2.GaussianBlur(inpaint_mask, (5, 5), 0) ,3,cv2.INPAINT_NS)
##        img_v = match_histograms(img_v, lighting_rgb[:, :, 1], multichannel=False)
#
#        cv2.imwrite('img_mh.png', img_v)
#        prob_mask_full[prob_mask_full<15] = 0
##        print(prob_mask_full)
#        blurred_mask = cv2.GaussianBlur(prob_mask_full, (21, 21), 11).astype('float32')/255.
#        print(np.amax(blurred_mask))
#        print("b")
#        cv2.imwrite('blurred_mask.png', 255*blurred_mask)
#        blurred_lighting = cv2.GaussianBlur(img_v, (21, 21), 11)
#        cv2.imwrite('blurred_lighting.png', blurred_lighting)
##        blurred_mask = match_histograms(blurred_mask, img_v, multichannel=False)
##        cv2.imwrite('blurred_mask.png', blurred_mask)
#        img_v = np.uint8(cv2.GaussianBlur(img_epf, (5, 5), 0)*(1.0-blurred_mask) + blurred_lighting*blurred_mask)
#        cv2.imwrite('img_v1.png', img_v)
##
#
#
#        lighting = match_histograms(img_v, lighting_rgb[:, :, 1], multichannel=False)
#        cv2.imwrite('img_v.png', img_v)
#        lighting = cv2.edgePreservingFilter(img_v, flags=1, sigma_s=30, sigma_r=0.4)
#        lighting = lighting_rgb[:, :, 1]

#
#inpaint_mask &= mask.astype('uint8')
#
#    inpaint_mask = cv2.dilate(inpaint_mask, kernel, iterations = 2)

    
    
#        smooth_lighting = cv2.inpaint(lighting.astype('uint8'), inpaint_mask.astype('uint8'),3,cv2.INPAINT_TELEA) # or INPAINT_TELEA
#        smooth_lighting = ip.remove_grooves(lighting, prob_mask_full)

#        blurred_mask = cv2.GaussianBlur(prob_mask_full, (31, 31), 15)
#        blurred_lighting = cv2.GaussianBlur(smooth_lighting, (21, 21), 11)

#        lighting = ip.alpha_blend(smooth_lighting, blurred_lighting, blurred_mask)

#        data["lighting"] = lighting
#        cv2.imwrite('lighting.png', lighting)
