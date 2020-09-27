from pipeline.core import PipelineStep
import cv2
import numpy as np
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.morphology import watershed, disk
from time import time
wall_like = [0, 8,14,18,22,42]
from cambrian import image_processing as ip
from skimage import filters, img_as_float
from skimage.filters import threshold_multiotsu, frangi
import os
import pickle

IM_LOGGING_ENABLED = False


def _log_image(name, image):
    if IM_LOGGING_ENABLED:
        cv2.imwrite('logging/' + name, image)


def _log_ply(image, masks, plane_XYZ, file_path='logging/3D.ply', write_occlusion=False, mult=1.0):
    if IM_LOGGING_ENABLED:
        image = cv2.resize(image, (mult * 160, mult * 120))
        width = image.shape[1]
        height = image.shape[0]
        faces = []
        points = []
        planes = np.zeros(shape=(len(masks), height, width, 3))
        new_masks = np.zeros(shape=(len(masks), height, width))
        for i in range(len(masks)):
            new_masks[i] = cv2.resize(masks[i], (width, height))
            planes[i] = cv2.resize(plane_XYZ[i], (width, height))
        plane_XYZ = planes
        masks = new_masks

        plane_depths = plane_XYZ[:, :, :, 1] * masks + 10 * (1 - masks)
        segmentation = np.argmin(plane_depths, axis=0)
        _log_image("logging/ply_segmention.png",cv2.normalize(segmentation, None, 0, 255, cv2.NORM_MINMAX))

        for mask_index, (mask, XYZ) in enumerate(zip(masks, plane_XYZ)):
            indices = np.nonzero(np.logical_and(segmentation == mask_index, masks[mask_index] > 0))
            for y, x in zip(indices[0], indices[1]):
                if y == height - 1 or x == width - 1:
                    continue
                validNeighborPixels = []
                for neighborPixel in [(x, y + 1), (x + 1, y), (x + 1, y + 1)]:
                    # if mask[neighborPixel[1], neighborPixel[0]] > 0.5:
                    validNeighborPixels.append(neighborPixel)
                    # pass
                    continue
                if len(validNeighborPixels) == 3:
                    faces.append([len(points) + c for c in range(3)])
                    points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for
                               pixel in
                               [(x, y), (x + 1, y + 1), (x + 1, y)]]
                    faces.append([len(points) + c for c in range(3)])
                    points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for
                               pixel in
                               [(x, y), (x, y + 1), (x + 1, y + 1)]]
                elif len(validNeighborPixels) == 2:
                    faces.append([len(points) + c for c in range(3)])
                    points += [(XYZ[pixel[1], pixel[0]], pixel, segmentation[pixel[1], pixel[0]] == mask_index) for
                               pixel in
                               [(x, y), (validNeighborPixels[0][0], validNeighborPixels[0][1]),
                                (validNeighborPixels[1][0], validNeighborPixels[1][1])]]
                    pass
                continue
            continue

        imageFilename = "textureless"
        with open(file_path, 'w') as f:
            header = """ply
        format ascii 1.0"""
            header += imageFilename
            header += """
        element vertex """
            header += str(len(points))
            header += """
        property float x
        property float y
        property float z
        property uchar red
        property uchar green
        property uchar blue
        element face """
            header += str(len(faces))
            header += """
        property list uchar int vertex_indices
        end_header
        """
            f.write(header)
            for point in points:
                X = point[0][0]
                Y = point[0][1]
                Z = point[0][2]
                if not write_occlusion or point[2]:
                    color = image[point[1][1], point[1][0]]
                else:
                    color = (128, 128, 128)
                    pass
                f.write(str(X) + ' ' + str(Z) + ' ' + str(-Y) + ' ' + str(color[2]) + ' ' + str(color[1]) + ' ' + str(
                    color[0]) + '\n')
                continue

            for face in faces:
                valid = True
                f.write('3 ')
                for c in face:
                    f.write(str(c) + ' ')
                    continue
                f.write('\n')
                continue
            f.close()
            pass
    return


a = 1.0
b = 1.0

METADATA = np.array([a * 571.87, b * 571.87, 320, 240, 640, 480, 0, 0, 0, 0])

IMAGE_MAX_DIM = 640
IMAGE_MIN_DIM = 480


def calcPlaneXYZ(planes, width=IMAGE_MIN_DIM, height=IMAGE_MAX_DIM, camera=METADATA, max_depth=10):
    urange = (np.arange(width, dtype=np.float32) / (width) * (camera[4]) - camera[2]) / camera[0]
    urange = urange.reshape(1, -1).repeat(height, 0)

    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera[5]) - camera[3]) / camera[1]
    vrange = vrange.reshape(-1, 1).repeat(width, 1)

    ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)

    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
    planeNormals = planes / np.maximum(planeOffsets, 1e-4)

    normalXYZ = np.dot(ranges, planeNormals.transpose())

    # _log_image("n.png", cv2.normalize(normalXYZ.transpose(2,0,1)[0], None, 0, 255, cv2.NORM_MINMAX))

    normalXYZ[normalXYZ == 0] = 1e-4

    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = np.clip(planeDepths, 0, max_depth)
        pass
    XYZ = (np.expand_dims(planeDepths, -1) * np.expand_dims(ranges, 2))

    return XYZ.transpose(2, 0, 1, 3), planeDepths.transpose(2, 0, 1)



def get_XYZ_from_depth(depth, width=IMAGE_MIN_DIM, height=IMAGE_MAX_DIM, camera=METADATA, max_depth=10):
    height = depth.shape[0]
    width = depth.shape[1]

    urange = (np.arange(width, dtype=np.float32) / (width) * (camera[4]) - camera[2]) / camera[0]
    urange = urange.reshape(1, -1).repeat(height, 0)

    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera[5]) - camera[3]) / camera[1]
    vrange = vrange.reshape(-1, 1).repeat(width, 1)

    X = depth * urange
    Y = depth
    Z = -depth * vrange
    XYZ = np.dstack((X, Y, Z))

    return XYZ


def get_segmentation_image(labels, image, avg=False, resize=True):
    if True:
        img_seg = image
        if resize:
            img_seg = cv2.resize(image, (labels.shape[1], labels.shape[0]))

        for label in range(1, np.amax(labels) + 1):
            color = np.random.randint([0, 0, 10], [254, 254, 235])
            if avg: color = np.mean(img_seg[labels == label], axis=0)
            img_seg[labels == label] = color
        return img_seg

def get_planes_class(plane_masks, class_labels):
    plane_classes = []

    indices = np.unique(class_labels)
    indices = indices[indices > -1]

    for i in range(len(plane_masks)):
        sums = ndimage.sum(plane_masks[i], labels=class_labels, index=indices)
        sorted = np.argsort(sums)[::-1]
        big = np.nonzero(sums[sorted]>np.max(sums)/2.)[0]
        plane_classes.append([indices[sorted[big]]])

    return plane_classes

def find_floor_indices(floor_mask, plane_masks, plane_normals):
    floor_mask = cv2.resize(floor_mask, (plane_masks[0].shape[1], plane_masks[0].shape[0]))

    floor_intersections = []
    dots = []
    for d in range(len(plane_masks)):
        floor_intersections.append(cv2.countNonZero(np.uint8(plane_masks[d][floor_mask > 1./3.] > .5)))

    scores = np.int32(floor_intersections)
    floor_indices = np.nonzero(scores > np.mean(scores))[0]
    floor_indices = floor_indices[np.argsort(scores[floor_indices])[::-1]]

    for d in range(len(plane_masks)):
        dots.append(np.dot(plane_normals[floor_indices[0]], plane_normals[d]))

    angs = np.arccos(np.clip(dots,-1.0,1.0))*180/np.pi

    horiz_indices = np.nonzero(np.abs(angs)<15)[0]
    return floor_indices, horiz_indices, angs


def find_wall_indices(wall_mask, plane_masks, floor_normal, plane_normals):
    wall_mask = cv2.resize(wall_mask, (plane_masks[0].shape[1], plane_masks[0].shape[0]))

    wall_intersections = []
    dots = []
    for d in range(len(plane_masks)):
        wall_intersections.append(cv2.countNonZero(np.uint8(plane_masks[d][wall_mask > 1./3.] > .5)))
        dots.append(np.dot(floor_normal, plane_normals[d]))

    angs = np.arccos(np.clip(dots, -1.0, 1.0)) * 180 / np.pi
    vert_indices = np.nonzero(np.abs(90 - angs) < 15)[0]

    scores = np.int32(wall_intersections)
    wall_indices = np.nonzero(np.logical_and(scores > np.mean(scores), np.abs(90 - angs) < 15))[0]

    return wall_indices, vert_indices, angs


def proj(points,plane):
    plane_offset = np.linalg.norm(plane)
    plane_normal = plane/plane_offset
    s = np.dot(points, plane_normal) - plane_offset
    return s

def compute_edgelets(image, lines, wall_mask = None):
    """Create edgelets as in the paper.
    Uses canny edge detection and then finds (small) lines using probabilstic
    hough transform as edgelets.
    Parameters
    ----------
    image: ndarray
        Image for which edgelets are to be computed.
    sigma: float
        Smoothing to be used for canny edge detection.
    Returns
    -------
    locations: ndarray of shape (n_edgelets, 2)
        Locations of each of the edgelets.
    directions: ndarray of shape (n_edgelets, 2)
        Direction of the edge (tangent) at each of the edgelet.
    strengths: ndarray of shape (n_edgelets,)
        Length of the line segments detected for the edgelet.
    """

    locations = []
    directions = []
    strengths = []

    for l in lines:

        l = l[0]
        p0, p1 = np.array([l[0],l[1]]), np.array([l[2],l[3]])
        c = (p0 + p1) / 2
        w = wall_mask[int(c[1]), int(c[0])]


        if w>.5:
            locations.append((p0 + p1) / 2)
            directions.append(p1 - p0)
            strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    directions = np.array(directions) / \
        np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths)


def edgelet_lines(edgelets):
    """Compute lines in homogenous system for edglets.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    Returns
    -------
    lines: ndarray of shape (n_edgelets, 3)
        Lines at each of edgelet locations in homogenous system.
    """
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines

import math
def compute_votes(edgelets, model, threshold_inlier=5, weight_mask=None):
    """Compute votes for each of the edgelet against a given vanishing point.
    Votes for edgelets which lie inside threshold are same as their strengths,
    otherwise zero.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    model: ndarray of shape (3,)
        Vanishing point model in homogenous cordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        edgelet direction and line connecting the  Vanishing point model and
        edgelet location is used to threshold.
    Returns
    -------
    votes: ndarry of shape (n_edgelets,)
        Votes towards vanishing point model for each of the edgelet.
    """
    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets

    est_directions = locations - vp

    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = np.abs(dot_prod / abs_prod)
    theta_thresh = np.cos(threshold_inlier * np.pi / 180)

    return (cosine_theta > theta_thresh) * strengths

def ransac_vanishing_point(edgelets, num_ransac_iter=2000, threshold_inlier=5, weight_mask=None, max_time =1.0, find_vert=True):
    """Estimate vanishing point using Ransac.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.
    Returns
    -------
    best_model: ndarry of shape (3,)
        Best model for vanishing point estimated.
    Reference
    ---------
    Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
    """
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)
    t = time()
    for ransac_iter in range(num_ransac_iter):
        if time() - t > max_time:
            return best_model

        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)


        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue

        if find_vert:
            if current_model[1]/current_model[2]<1000: continue

            dt1 = abs(np.dot(directions[ind1], [0,1]))
            dt2 = abs(np.dot(directions[ind2], [0,1]))

            if dt1 <.95 or dt2<.95:
                continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier, weight_mask)

        if current_votes.sum() > best_votes.sum():

            best_model = current_model
            best_votes = current_votes
            # print("Current best model has {} votes at iteration {},{}".format(
            #     current_votes.sum(), ransac_iter, np.round(time()-t,2)))


    return best_model


class PipelineRefinePlaneMasks(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "hed", "mask"]

    @property
    def output_keys(self) -> list:
        return ["planes"]

    def run(self, data):

        img = data["image"]

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        output = data["semantic_probs"]

        h, w = output[0].shape


        shape = (w, h)
        img_rs = cv2.resize(img, shape)

        _log_image('img.png', img_rs)

        hed = data["hed"]
        hed_rs = cv2.resize(hed, shape)
        _log_image('hed.png', hed_rs)

        planes_data = data["planes"]


        # merge rugs into floor for time being
        rug = output[28]
        output[3] += rug
        output[28] = 0

        wall_mask = output[0]
        _log_image("wall_mask.png", 255. * wall_mask)

        floor_mask = output[3]
        _log_image("floor_mask.png", 255. * wall_mask)

        output_p = np.zeros((151, output.shape[1], output.shape[2]))

        output_p[1:152] = output
        output_p[0] = hed_rs / 255.

        predict = np.int32(np.argmax(output_p, 0))

        # initial watershed to clean things up a bit
        predict = cv2.watershed(img_rs, markers=predict) - 1
        # _log_image("semantic_seg_initial.png", get_segmentation_image(predict, img_rs, avg=False))

        length_threshold = 16
        canny_aperture_size = 7

        fld = cv2.ximgproc.createFastLineDetector(_length_threshold=length_threshold,
                                                  _canny_aperture_size=canny_aperture_size)
        lines = fld.detect(cv2.cvtColor(img_rs, cv2.COLOR_BGR2GRAY))

        plane_masks = planes_data["masks"]
        number_planes = len(plane_masks)

        plane_XYZ = planes_data["plane_XYZ"][:, :, 80:-80, :].transpose(0, 2, 3, 1)
        # _log_ply(img_rs, plane_masks, plane_XYZ, mult=1, file_path='3D.ply')

        plane_parameters = np.array(data["planes"]["detection"][:, 6:9], dtype=np.float32)
        plane_offsets = np.linalg.norm(plane_parameters, axis=-1, keepdims=True)
        plane_normals = plane_parameters / np.maximum(plane_offsets, 1e-4)
        plane_clusters = np.array(data["planes"]["detection"][:, 4], dtype=np.int32)
        roi = np.array(data["planes"]["detection"][:, 0:4], dtype=np.int32)

        for i in range(number_planes):
            box = roi[i].copy()
            box[0] = np.clip(box[0], 0, 479)
            box[1] = np.clip(box[1], 0, 639)
            box[2] = np.clip(box[2], 0, 479)
            box[3] = np.clip(box[3], 0, 639)
            roi[i] = box

        floor_indices, horiz_indices, floor_angs = find_floor_indices(floor_mask, plane_masks, plane_normals)

        # print("floor indices are: ", floor_indices, horiz_indices)

        floor_index = floor_indices[0]

        floor_normal = plane_normals[floor_index]
        wall_indices, vert_indices, vert_angs = find_wall_indices(wall_mask, plane_masks, plane_normals[floor_index],
                                                                    plane_normals)
        # print("wall indices are: ", wall_indices, vert_indices, vert_angs)

        for i in range(number_planes):
            if i in vert_indices:
                mass = np.sum(plane_masks[i])
                if mass > 0:
                    plane_normals[i] = np.cross(floor_normal, np.cross(floor_normal, plane_normals[i]))

                    plane_parameters[i] = plane_normals[i] * np.sum(
                        plane_masks[i] * np.dot(plane_XYZ[i], plane_normals[i])) / mass

        plane_XYZ, plane_depth = calcPlaneXYZ(plane_parameters, width=640, height=480, max_depth=10)
        #
        for i in np.unique(plane_clusters):

            cluster = np.nonzero(plane_clusters == i)[0]

            for j in cluster:
                if j not in vert_indices: continue

                for k in cluster:
                    if k <= j: continue
                    rect = roi[j]
                    plane_rect = [plane_XYZ[j, rect[0], rect[1]], plane_XYZ[j, rect[0], rect[3]],
                                  plane_XYZ[j, rect[2], rect[3]], plane_XYZ[j, rect[2], rect[1]]]
                    p = np.abs(proj(plane_rect, plane_parameters[k]))

                    if len(p) > 0 and (np.mean(p) < .3):
                        plane_parameters[k] = plane_parameters[j]

        plane_XYZ, plane_depth = calcPlaneXYZ(plane_parameters, width=640, height=480, max_depth=10)

        plane_masks_rs = np.zeros((number_planes, shape[1], shape[0]))
        for k in range(number_planes):
            plane_masks_rs[k] = cv2.resize(plane_masks[k], shape)

        plane_masks = plane_masks_rs

        # everything but floor and ceiling
        edgelet_mask = np.uint8(predict != 3)
        edgelet_mask[predict == 5] = 0

        edgelets = compute_edgelets(img_rs, lines, edgelet_mask)

        vp0 = ransac_vanishing_point(edgelets, 3000, threshold_inlier=3, max_time=0.2)
        vp0 = vp0 / vp0[2]
        vertical_line_inliers = compute_votes(edgelets, vp0, 2) > 0
        locations, directions, strengths = edgelets
        edgelets = (
        locations[vertical_line_inliers], directions[vertical_line_inliers], strengths[vertical_line_inliers])
        locations, directions, strengths = edgelets

        img_lines = np.int32(np.zeros_like(plane_masks[0]))
        img_fan = np.int32(np.zeros_like(plane_masks[0]))

        for i in range(locations.shape[0]):
            # s = (vp0[:2]-locations[i])*directions[i]
            xax = [locations[i, 0] + (locations[i, 0] - vp0[0]), vp0[0]]
            yax = [locations[i, 1] + (locations[i, 1] - vp0[1]), vp0[1]]

            m = wall_mask[int(locations[i, 1]), int(locations[i, 0])]

            if m > .01:
                cv2.line(img_fan, (int(xax[0]), int(yax[0])), (int(xax[1]), int(yax[1])), color=i + 1, thickness=1,
                         lineType=cv2.LINE_4)

            xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2.,
                   locations[i, 0] + directions[i, 0] * strengths[i] / 2.]
            yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2.,
                   locations[i, 1] + directions[i, 1] * strengths[i] / 2.]
            cv2.line(img_lines, (int(xax[0]), int(yax[0])), (int(xax[1]), int(yax[1])), color=i + 1, thickness=1,
                     lineType=cv2.LINE_4)
            cv2.line(img_fan, (int(xax[0]), int(yax[0])), (int(xax[1]), int(yax[1])), color=i + 1, thickness=1,
                     lineType=cv2.LINE_4)

        # _log_image("fan_lines.png", get_segmentation_image(img_fan, img_rs, avg=False))
        # _log_image("img_lines.png", get_segmentation_image(img_lines, img_rs, avg=False))

        floor_mask = output[3]
        floor_mask[predict != 3] = 0
        _log_image("floor_mask.png", 255. * floor_mask)

        plane_classes = get_planes_class(plane_masks, predict)

        plane_masks[plane_masks < .05] = 0
        wall_like_planes = [hed_rs / 255.]
        wall_like_depths = [10. * np.ones_like(plane_masks[0])]
        wall_like_indices = []
        wall_planes = [np.zeros_like(plane_masks[0])]
        wall_indices = []
        floor_indices = []
        ceiling_indices = []

        for k in range(number_planes):
            ade = np.int32(plane_classes[k])
            name = "wall_plane" + str(k)

            for a in ade[0]:

                _log_image(name + ".png", 255. * plane_masks[k])
                if a == 0:
                    wall_planes.append(plane_masks[k])
                    wall_indices.append(k)

                if a in wall_like:
                    wall_like_planes.append(plane_masks[k])
                    wall_like_indices.append(k)
                    s = np.uint8(plane_masks[k] > 0)
                    wall_like_depths.append(cv2.resize(plane_XYZ[k, :, :, 1],shape) * s + (1 - s) * 10)
                    # _log_image(name + ".png", 255. * plane_masks[k])
                    continue

                if a == 3:
                    floor_indices.append(k)
                    continue

                if a == 5:
                    ceiling_indices.append(k)
                    continue

        wall_like_planes = np.float32(wall_like_planes)
        wall_planes = np.float32(wall_planes)
        wall_like_depths = np.float32(wall_like_depths)
        planes_seg = np.argmax(wall_like_planes, 0)
        planes_seg = np.argmin(np.float32(wall_like_depths), 0)
        for label in np.unique(planes_seg):
            mask = planes_seg == label
            components, component_labels, stats, centroids = cv2.connectedComponentsWithStats(np.uint8(mask),
                                                                                              connectivity=4)
            sizes = stats[:, -1][1:]

            small = np.nonzero(sizes < np.max(sizes / 2.))[0] + 1
            for s in small:
                planes_seg[component_labels == s] = 0

        # _log_image("planes_wall_segmentation.png",  get_segmentation_image(planes_seg, img_rs, avg=False))


        wall_mask[predict != 0] = 0
        wall_mask_edges = wall_mask.copy()
        wall_mask_edges[img_fan > 0] = 0

        _log_image("wall_mask_edges.png", 255. * wall_mask_edges)

        distances = cv2.distanceTransform(np.uint8(wall_mask_edges > 0), cv2.DIST_L2, 5)
        markers = ndimage.label(distances > 1)[0]

        # _log_image("labels_wall_pre.png", get_segmentation_image(markers, img_rs, avg=False))
        local_maxi = peak_local_max(distances, labels=markers, num_peaks_per_label=1, indices=False)

        markers = ndimage.label(local_maxi)[0]

        markers[markers > 0] += number_planes
        markers[np.logical_and(markers > 0, planes_seg > 0)] = planes_seg[np.logical_and(markers > 0, planes_seg > 0)]

        labels_wall = np.int32(watershed(-distances, markers=markers, mask=np.uint8(wall_mask > .5)))

        # _log_image("labels_wall_watershed.png", get_segmentation_image(np.uint8(labels_wall), img_rs, avg=False))

        unknown = np.unique(labels_wall[labels_wall > number_planes])

        wall_planes_number = np.amax(np.unique(planes_seg))

        means = np.zeros((len(unknown), wall_planes_number + 1), dtype=np.float32)
        for i in range(1, wall_planes_number + 1):
            means[:, i] = ndimage.mean(np.uint8(planes_seg == i), labels=labels_wall, index=unknown)

        # means[means<.03] = 0
        arg = np.argmax(means, axis=-1)

        labels_arg = np.int32(labels_wall.copy())

        for i in range(len(unknown)):
            labels_arg[labels_wall == unknown[i]] = arg[i]

        # labels_wall[labels_wall>number_planes] = 0
        # _log_image('watershed_arg.png', get_segmentation_image(labels_arg, img_rs, avg=False))

        labels_wall = labels_arg + 1
        labels_wall[wall_mask < hed_rs / 255.] = 0

        labels_wall = watershed(-distances, markers=labels_wall, mask=np.uint8(predict == 0)) - 1
        labels_wall[labels_wall < 1] = 0
        # _log_image("labels_wall.png", get_segmentation_image(np.uint8(labels_wall), img_rs, avg=False))


        final_masks = []
        final_plane_parameters = []
        final_plane_XYZ = []

        for l in np.unique(labels_wall):
            if l == 0: continue

            mask = np.uint8(labels_wall == l)
            final_masks.append(mask)
            final_plane_parameters.append(plane_parameters[wall_like_indices[l - 1]])
            final_plane_XYZ.append(plane_XYZ[wall_like_indices[l - 1]])
            _log_image('final' + str(l) + "f.png", 255. * mask)

        # add floor
        final_masks.append(predict == 3)
        final_plane_parameters.append(plane_parameters[floor_indices[0]])
        final_plane_XYZ.append(plane_XYZ[floor_indices[0]])

        # add ceiling
        if len(ceiling_indices) > 0:
            final_masks.append(predict == 5)
            final_plane_parameters.append(plane_parameters[ceiling_indices[0]])
            final_plane_XYZ.append(plane_XYZ[ceiling_indices[0]])

        final_plane_parameters = np.float32(final_plane_parameters)
        final_masks = np.uint8(final_masks)

        final_masks = []
        final_plane_parameters = []
        final_plane_XYZ = []

        for l in np.unique(labels_wall):
            if l == 0: continue

            mask = 255*np.uint8(labels_wall == l)
            final_masks.append(mask)
            plane_parameter = np.zeros((10))
            plane_parameter[:9] = data["planes"]["detection"][wall_like_indices[l - 1]][:9]
            plane_parameter[6:9] = plane_parameters[wall_like_indices[l - 1]]
            plane_parameter[9] = 2
            final_plane_parameters.append(plane_parameter)
            final_plane_XYZ.append(plane_XYZ[wall_like_indices[l - 1]])
            _log_image('final' + str(l) + ".png", 255. * mask)

        # add floor
        if len(floor_indices) > 0:
            final_masks.append(255*np.uint8(predict == 3))
            plane_parameter = np.zeros((10))
            plane_parameter[:9] = data["planes"]["detection"][floor_indices[0]][:9]
            plane_parameter[6:9] = plane_parameters[floor_indices[0]]
            plane_parameter[9] = 1
            final_plane_parameters.append(plane_parameter)
            final_plane_XYZ.append(plane_XYZ[floor_indices[0]])

        # add ceiling
        if len(ceiling_indices) > 0:
            final_masks.append(255*np.uint8(predict == 5))
            plane_parameter = np.zeros((10))
            plane_parameter[:9] = data["planes"]["detection"][ceiling_indices[0]][:9]
            plane_parameter[6:9] = plane_parameters[ceiling_indices[0]]
            plane_parameter[9] = 3
            final_plane_parameters.append(plane_parameter)
            final_plane_XYZ.append(plane_XYZ[ceiling_indices[0]])

        final_plane_parameters = np.float32(final_plane_parameters)

        # final_plane_XYZ = np.float32(final_plane_XYZ)

        final_plane_number = len(final_masks)

        print("final_plane_number", final_plane_number)

        data["planes"]["masks"] = np.zeros((final_plane_number, shape[1], shape[0]), dtype=np.uint8)

        data["planes"]["detection"] = np.zeros((final_plane_number, 10), dtype=data["planes"]["detection"].dtype)
        data["planes"]["detection"] = final_plane_parameters

        mask_contours = []

        for d in range(final_plane_number):
            contours, hierarchy = cv2.findContours(final_masks[d], cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)
            plane_contours = []
            if len(contours) > 0:

                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])

                    if area > 16*16:
                        if hierarchy[0, i, 3] == -1:  # this is the outer contour which we need to draw
                            cv2.drawContours(data["planes"]["masks"][d], [contours[i]], -1, 255, -1)

                            epsilon = cv2.arcLength(contours[i], True) / shape[0]
                            approx = cv2.approxPolyDP(contours[i], epsilon, closed=True)
                            plane_contours.append(approx.tolist())
                        else:
                            cv2.drawContours(data["planes"]["masks"][d], contours, i, 0, -1)

                    final_mask = cv2.GaussianBlur(np.uint8(data["planes"]["masks"][d]), (5, 5), 0)

                    _, data["planes"]["masks"][d] = cv2.threshold(
                        final_mask, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                rect = cv2.boundingRect(data["planes"]["masks"][d])

                data["planes"]["detection"][d, 0:4] = [
                    rect[1], rect[0], rect[1] + rect[3], rect[0] + rect[2]
                ]

            mask_contours.append(plane_contours)

        data["planes"]["contours"] = mask_contours

        # _log_ply(img_rs, data["planes"]["masks"], final_plane_XYZ, mult=2, file_path='logging/3D_refine.ply')

