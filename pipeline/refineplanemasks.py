from pipeline.core import PipelineStep
import cv2
import numpy as np
from scipy import ndimage
from time import time
from skimage.morphology import skeletonize
from cambrian import image_processing as ip
from cambrian import frei_chen, Line

import os
import pickle
import math
from skimage.segmentation import watershed
from scipy.stats import mode
from skimage.morphology import remove_small_objects, remove_small_holes

logging_dir ='logging/'
#
# for file in os.scandir(logging_dir):
#     if file.name.endswith(".png"):
#         os.remove(file)


IM_LOGGING_ENABLED = True

furniture_labels = [15, 23, 30, 64, 97]
wall_like = [0, 8, 14, 18, 22, 24, 42, 58,63, 130]
wall_int = [3, 8, 22, 100]


def _log_image(name, image, logging_index = 0):
    if IM_LOGGING_ENABLED:
        name = str(logging_index) + " - " + name
        cv2.imwrite(logging_dir + name, image)
    return logging_index + 1

def _log_segmentation_image(name, segmentation, image, avg=False, logging_index=0):
    if IM_LOGGING_ENABLED:
        name = str(logging_index) + " - " + name
        seg = get_segmentation_image(segmentation, image, avg)
        cv2.imwrite(logging_dir + name, seg)
    return logging_index + 1


def _log_ply(image, masks, plane_XYZ, file_path=logging_dir + '3D.ply', write_occlusion=False, mult=1.0):
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

        for mask_index, (mask, XYZ) in enumerate(zip(masks, plane_XYZ)):
            indices = np.nonzero(np.logical_and(segmentation == mask_index, masks[mask_index] > 0))
            indices = np.nonzero(masks[mask_index] > .05)
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
        # big = np.nonzero(sums[sorted] > np.max(sums) / 2.)[0]
        plane_classes.append(indices[sorted[0]])

    return plane_classes


def find_floor_indices(floor_mask, ceiling_mask, plane_masks, plane_normals):
    # floor_mask = cv2.resize(floor_mask, (plane_masks[0].shape[1], plane_masks[0].shape[0]))

    floor_intersections = []
    ceiling_intersections = []

    dots = []
    for d in range(len(plane_masks)):
        floor_intersections.append(cv2.countNonZero(
            np.uint8(plane_masks[d][floor_mask > np.amax(floor_mask) / 2.] > np.amax(plane_masks[d]) / 2.)))
        ceiling_intersections.append(cv2.countNonZero(
            np.uint8(plane_masks[d][ceiling_mask > np.amax(ceiling_mask) / 2.] > np.amax(plane_masks[d]) / 2.)))

    scores = np.int32(floor_intersections)
    floor_indices = np.nonzero(scores > np.mean(scores))[0]
    floor_indices = floor_indices[np.argsort(scores[floor_indices])[::-1]]

    scores = np.int32(ceiling_intersections)
    ceiling_indices = np.nonzero(scores > np.mean(scores))[0]
    ceiling_indices = ceiling_indices[np.argsort(scores[ceiling_indices])[::-1]]

    for d in range(len(plane_masks)):
        dots.append(np.dot(plane_normals[floor_indices[0]], plane_normals[d]))

    angs = np.arccos(np.clip(dots, -1.0, 1.0)) * 180 / np.pi

    horiz_indices = np.nonzero(np.abs(angs) < 15)[0]
    return floor_indices, ceiling_indices, horiz_indices, angs


def find_wall_indices(wall_mask, plane_masks, floor_normal, plane_normals):
    wall_mask = cv2.resize(wall_mask, (plane_masks[0].shape[1], plane_masks[0].shape[0]))

    wall_intersections = []
    dots = []
    for d in range(len(plane_masks)):
        wall_intersections.append(cv2.countNonZero(np.uint8(plane_masks[d][wall_mask > 1. / 3.] > .5)))
        dots.append(np.dot(floor_normal, plane_normals[d]))

    angs = np.arccos(np.clip(dots, -1.0, 1.0)) * 180 / np.pi
    vert_indices = np.nonzero(np.abs(90 - angs) < 15)[0]

    scores = np.int32(wall_intersections)
    wall_indices = np.nonzero(np.logical_and(scores > np.mean(scores), np.abs(90 - angs) < 15))[0]

    return wall_indices, vert_indices, angs

def proj(points, plane):
    plane_offset = np.linalg.norm(plane)
    plane_normal = plane / plane_offset
    s = np.dot(points, plane_normal) - plane_offset
    return s



def compute_edgelets(lines, class_labels=[]):
    """Create edgelets as in the paper.
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
    classes = []

    check_class = False
    if len(class_labels)>0:
        check_class = True

    for l in lines:
        l = l[0]
        p0, p1 = np.array([l[0], l[1]]), np.array([l[2], l[3]])
        c = (p0 + p1) / 2
        if check_class:
            classes.append(class_labels[int(c[1]), int(c[0])])

        locations.append(c)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)
    classes = np.array(classes)

    directions = np.array(directions) / \
                 np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths, classes)


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
    locations, directions = edgelets[:2]
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines


def compute_votes(edgelets, model, threshold_inlier=5, model_indices=[-1,-1]):
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

    locations, directions, strengths = edgelets[:3]

    est_directions = locations - vp

    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1)*np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = np.abs(dot_prod / abs_prod)

    theta_thresh = np.cos(threshold_inlier * np.pi / 180)
    good = (cosine_theta > theta_thresh)

    return good



def ransac_vanishing_point(edgelets, num_ransac_iter=2000, threshold_inlier=5, max_time=1.0, class_labels=[]):
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
    locations, directions, strengths, classes = edgelets

    angles = np.arctan2(np.sign(directions[:,1])*directions[:, 1], np.sign(directions[:,1])*directions[:, 0])

    lines = edgelet_lines(edgelets)
    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)

    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    class_number = len(np.unique(class_labels))
    class_number=1

    best_models = np.zeros((class_number+1, 3))
    best_votes = np.zeros((class_number+1), np.int32)

    t = time()

    for ransac_iter in range(num_ransac_iter):
        if time() - t > max_time:
            return best_models, best_votes

        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        if ind1==ind2: continue

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue

        good = compute_votes(
            edgelets, current_model, threshold_inlier)
        current_votes = (good*strengths).sum()

        dt1 = abs(np.dot(directions[ind1], [0, 1]))
        dt2 = abs(np.dot(directions[ind2], [0, 1]))

        if dt1 > .95 and dt2 > .95 and abs(current_model[1]/current_model[2])>1000:
            if current_votes > best_votes[0]:
                best_models[0] = current_model
                best_votes[0] = current_votes
                # print("angles", angles[ind1], angles[ind2], dt1, dt2)
                # print("vertical", ransac_iter, current_number, directions[ind1], directions[ind2], np.int32(current_model/current_model[2]))
            continue

        continue
        # if class1!=class2: continue

        if current_votes > best_votes[1]:
            best_models[1] = current_model
            best_votes[1] = current_votes

    return best_models, best_votes


def calcTransformation(points_1, points_2):
    # center_1 = points_1.mean(0) =(0,0,0)
    # center_2 = points_2.mean(0)=(0,0,0)
    center_1 = (0, 0, 0)
    center_2 = (0, 0, 0)
    H = np.matmul((points_1 - center_1).transpose(), (points_2 - center_2))
    U, S, V = np.linalg.svd(H)

    R = np.matmul(V.transpose(), U.transpose())
    if np.linalg.det(R) < 0 and False:
        R[:, 2] *= -1
        pass
    t = -np.matmul(R, center_1) + center_2
    return R, t



def combined_normals(normals, plane_normals, plane_masks, basis_indices, cluster_prob):
    plane_normals_nn = np.zeros_like(plane_normals)
    number_planes = len(plane_normals)
    cluster_indices = cluster_prob > .5
    cluster_indices = cluster_prob > .5
    print("good clusters", cluster_indices)
    basis_indices = basis_indices[cluster_indices[basis_indices]]

    for i in range(number_planes):
        if cluster_indices[i]:
            plane_normals_nn[i] = mode(normals[plane_masks[i] > np.amax(plane_masks[i]) / 2.], axis=0)[0]
            plane_normals_nn[i] /= np.linalg.norm(plane_normals_nn[i])

    R, _ = calcTransformation(plane_normals_nn[basis_indices], plane_normals[basis_indices])

    nr = normals.reshape((-1, 3))
    normals = np.matmul(R, nr.transpose()).transpose().reshape(normals.shape)

    lengths = np.maximum(np.sqrt(np.sum(normals * normals, -1)), 1e-6)
    normals /= np.dstack((lengths, lengths, lengths))
    # plane_normals_nn = np.matmul(R,plane_normals_nn.transpose()).transpose()

    for i in range(number_planes):
        # if cluster_indices[i]:
        m = plane_masks[i].copy()
        # m[m>.25] = 1
        mult = np.dstack((m,m,m))

        normals = (1.0 - mult) * normals + mult * plane_normals[i]

    lengths = np.maximum(np.sqrt(np.sum(normals * normals, -1)), 1e-6)
    normals /= np.dstack((lengths, lengths, lengths))
    return normals


def refine_surface(mask, image, big_thresh=.03, small_thresh=.97, watershed_dist=.05, watershed_mask=None, gradient=True):
    small = ip.refine_mask_watershed(None, image, np.uint8(mask > small_thresh), None, distance=watershed_dist, gradient=gradient,
                                   watershed_mask=watershed_mask)

    big = ip.refine_mask_watershed(None, image, np.uint8(mask > big_thresh), None, distance=watershed_dist, gradient=gradient,
                                     watershed_mask=watershed_mask)

    markers = np.dstack((np.ones_like(big), small, big))
    markers = np.argmax(markers, -1)

    markers[markers == 1] = (ndimage.label(markers == 1)[0])[markers == 1] + np.amax(markers)
    markers[markers == 2] = (ndimage.label(markers == 2)[0])[markers == 2] + np.amax(markers)

    return markers



def calcTransformation(points_1, points_2):
    # center_1 = points_1.mean(0) =(0,0,0)
    # center_2 = points_2.mean(0)=(0,0,0)
    center_1 = (0, 0, 0)
    center_2 = (0, 0, 0)
    H = np.matmul((points_1 - center_1).transpose(), (points_2 - center_2))
    U, S, V = np.linalg.svd(H)

    R = np.matmul(V.transpose(), U.transpose())
    if np.linalg.det(R) < 0 and False:
        R[:, 2] *= -1
        pass
    t = -np.matmul(R, center_1) + center_2
    return R, t


def resize_array(array, shape):
    length = len(array)
    print(array.shape)

    dim=1
    array_rs = np.zeros((length, shape[1], shape[0]))

    if array[0].ndim>2:
        dim = array.shape[-1]
        array_rs = np.zeros((length, shape[1], shape[0], dim))

    for k in range(length):
        array_rs[k] = cv2.resize(array[k], shape)

    return array_rs


def rough_dilate_erode(is_dilate, mask, size=5, iterations=1, scale=0.5, maintain_size=True, interpolation=cv2.INTER_NEAREST):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    shape = mask.shape
    mask = cv2.resize(mask, (int(shape[1] * scale), int(shape[0] * scale)), interpolation)
    mask = cv2.dilate(mask, kernel, iterations=iterations) if is_dilate else cv2.erode(mask, kernel, iterations=iterations)
    if maintain_size:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation)
    return mask


def find_lines(img, gradient, normals, mask, output_path, contour_masks = []):
    line_time = time()

    height, width = img.shape[:2]
    diagonal = np.hypot(width, height)
    print("image w,h", width, height)

    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gabor_scale = 1500.0 / diagonal
    bw_res = cv2.resize(bw, (int(width * gabor_scale), int(height * gabor_scale)),
                        cv2.INTER_CUBIC) if gabor_scale < 1.0 else bw

    def gabor(theta, lambd, gamma=0.0, psi=0.0):
        ksize = lambd
        sigma = ksize * lambd
        result = cv2.filter2D(bw_res, cv2.CV_8UC1,
                              cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F))
        return result

    v_gabor = gabor(0, 7)
    h_gabor = gabor(np.pi / 2.0, 9)

    contours_src = cv2.addWeighted(v_gabor, 1.0, h_gabor, 1.0, 0)
    contours_src = cv2.resize(contours_src, (width, height), interpolation=cv2.INTER_CUBIC)

    edges = cv2.addWeighted(v_gabor, 3.0, h_gabor, 3.0, -20)
    edges = cv2.bilateralFilter(edges, 5, 5, 5)
    edges = cv2.resize(edges, (width, height), interpolation=cv2.INTER_CUBIC)

    clean_edges = frei_chen(bw)

    line_data = []

    lines_c = img.copy()


    def is_image_edge(point_a, point_b, shape, dist=10):
        max_0 = shape[0] - 1
        max_1 = shape[1] - 1
        return (abs(point_a[0]) <= dist and abs(point_b[0]) <= dist) \
               or (abs(point_a[0] - max_0) <= dist and abs(point_b[0] - max_0) <= dist) \
               or (abs(point_a[1]) <= dist and abs(point_b[1]) <= dist) \
               or (abs(point_a[1] - max_1) <= dist and abs(point_b[1] - max_1) <= dist)

    def add_contour_lines(contours, min_confidence):
        epsilon = diagonal / 200.0
        min_length = diagonal / 40.0
        contour_group = 0
        for contour in contours:

            poly = cv2.approxPolyDP(contour, epsilon, False)
            contour_index = 0
            arcLen = cv2.arcLength(poly, False)

            for i in range(0, len(poly) - 1):
                point_a = poly[i][0]
                point_b = poly[i + 1][0]

                # remove contours on image edge and break them up into seperate contours (contour_group)
                if is_image_edge(point_a, point_b, (width, height), epsilon + 1.0):
                    contour_index = 0
                    contour_group += 1
                elif arcLen > min_length:
                    new_line = Line(point_a[0], point_a[1], point_b[0], point_b[1], contour_group, contour_index)
                    line_data.append(new_line)
                    contour_index += 1

        contour_group += 1

    contours_src = cv2.adaptiveThreshold(contours_src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                         int(diagonal / 50) * 2 + 1, -30)
    contours_dilated = rough_dilate_erode(True, contours_src, 3, scale=400 / diagonal, interpolation=cv2.INTER_AREA)
    if gradient is None:
        Line.prepare(img, contours_dilated, lines_c)
    else:
        Line.prepare(np.dstack((img, gradient)), contours_dilated, lines_c)

    # find all liens in the edge image
    fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 60.0), 1.41, 200, 240, 3, False)
    lines1 = fld.detect(edges)

    aperture = 5
    fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 30.0), 1.41, 200, 220, aperture, False)
    lines2 = fld.detect(bw - (clean_edges * 5.0).astype("uint8"))

    aperture = 5
    fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 15.0), 1.41,_canny_aperture_size=aperture, _do_merge=False)
    lines3 = fld.detect(edges)

    sy = edges.shape[0] / normals.shape[0]
    sx = edges.shape[1] / normals.shape[1]
    fld = cv2.ximgproc.createFastLineDetector(64, _canny_aperture_size=7, _do_merge=False)
    lines4 = fld.detect(cv2.cvtColor(np.uint8(normals), cv2.COLOR_BGR2GRAY))
    lines4 = lines4 * [[sx, sy, sx, sy]]

    lines = np.concatenate((lines1, lines2, lines3, lines4))
    confs = []

    if lines is not None:
        for line in lines:
            new_line = Line(line[0][0], line[0][1], line[0][2], line[0][3])
            # mp = np.int32(new_line.midpoint)
            # pt1 = np.int32(new_line.point_a)
            # pt2 = np.int32(new_line.point_b)
            cm = new_line.get_color_mean()
            print("color_mean", cm)
            if cm[3]<15: continue

            # if mask[mp[1],mp[0]]>0: continue
            conf = new_line.get_confidence()
            if conf > 0.2:
                confs.append(conf)
                line_data.append(new_line)

    line_data = [line_data[i] for i in np.argsort(confs)]

    line_data = Line.merge(line_data, diagonal / 120.0, search_length=1.03, angle_threshold=math.radians(3.0))
    # for line in lines:
    #     new_line = Line(line[0][0], line[0][1], line[0][2], line[0][3])
    #     # if mask[mp[1],mp[0]]>0: continue
    #     conf = new_line.get_confidence()
    #     if conf > 0.8:
    #         line_data.append(new_line)
    # line_data = Line.merge(line_data, diagonal / 120.0, search_length=1.01, angle_threshold=math.radians(3.0))

    # line_data = Line.merge(line_data, diagonal /90.0, search_length=1.03, angle_threshold=math.radians(2.0))

    #
    # line_data = Line.merge(line_data, diagonal / 90.0, search_length=1.0, angle_threshold=math.radians(3.0),
    #                        color=(255, 0, 0))
    # # # #
    line_data, intersections = Line.find_corners(line_data, search_length=1.5, angle_threshold=math.radians(5.0),
                                                 parallel_threshold=math.radians(5), confidence_diff=0.4)
    # # #
    # line_data = Line.merge(line_data, diagonal / 200.0, search_length=1.07, angle_threshold=math.radians(5.0),
    #                        color=(255, 0, 0))
    # print("find lines time", time() - line_time)

    return line_data, lines



class PipelineRefinePlaneMasks(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "hed", "mask"]

    @property
    def output_keys(self) -> list:
        return ["planes"]

    def run(self, data):

        if IM_LOGGING_ENABLED:
            with open('logging/data.pickle', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        img = data["image"]

        logging_index = _log_image("image.png", img, logging_index=logging_index)

        output = np.float32(data["semantic_probs"])
        org_time = time()

        hed = data["hed"]

        hed_rs = hed
        logging_index = _log_image('hed.png', hed_rs, logging_index=logging_index)

        h, w = output[0].shape

        shape = (w, h)

        hed_rs = cv2.resize(hed, shape)

        img_rs = cv2.resize(img, shape)

        sx = w / img.shape[1]
        sy = h / img.shape[0]

        rug = output[28]
        output[3] += rug
        output[28] = 0
        wall_mask = output[0].copy()

        logging_index = _log_image("wall_mask.png", 255. * (wall_mask), logging_index=logging_index)

        floor_mask = output[3].copy()
        logging_index = _log_image("floor_mask.png", 255. * floor_mask, logging_index=logging_index)

        ceiling_mask = output[5].copy()
        logging_index = _log_image("ceiling_mask.png", 255. * ceiling_mask, logging_index=logging_index)

        wall_like_mask = np.zeros_like(wall_mask)

        for i in wall_like:
            if i == 0: continue
            wall_like_mask += output[i]

        logging_index = _log_image("wall_like_mask.png", 255. * wall_like_mask, logging_index=logging_index)

        other_mask = 1.0 - floor_mask - wall_mask - wall_like_mask - ceiling_mask

        logging_index = _log_image("other_mask.png", 255. * other_mask, logging_index=logging_index)

        ade_seg_c = np.dstack(
            (.9 * np.ones_like(other_mask), other_mask, floor_mask, wall_mask, ceiling_mask, wall_like_mask))

        ade_seg = np.argmax(ade_seg_c, -1)

        logging_index = _log_segmentation_image("ade_seg.png", np.int32(ade_seg), img_rs, logging_index=logging_index)

        line_data, lines = find_lines(img, cv2.resize(hed, (img.shape[1], img.shape[0])), data["normals"],
                                      cv2.resize(np.uint8(ade_seg > 0), (img.shape[1], img.shape[0])), logging_dir)

        all_lines = np.int32(np.zeros((img_rs.shape[0], img_rs.shape[1])))

        l_image_rgb = img_rs.copy()
        l = 1

        for line in lines:
            for x1, y1, x2, y2 in line:
                x1 = int(sx * x1)
                x2 = int(sx * x2)
                y1 = int(sy * y1)
                y2 = int(sy * y2)
                cv2.line(all_lines, (x1, y1), (x2, y2), l, thickness=2, lineType=cv2.LINE_8)
                l += 1

        merged_lines = np.int32(np.zeros((img_rs.shape[0], img_rs.shape[1])))
        Line.draw_all(line_data, merged_lines, color=255, thickness=2, sx=sx, sy=sy)

        l_image_rgb[merged_lines > 0] = 255
        logging_index = _log_segmentation_image("l_image.png", all_lines, img_rs, logging_index=logging_index)

        logging_index = _log_image("l_image_rgb.png", l_image_rgb, logging_index=logging_index)

        edgelets = compute_edgelets(lines, class_labels=[])

        vanishingpts, votes = ransac_vanishing_point(edgelets, 5000, threshold_inlier=2, max_time=.5,
                                                     class_labels=[])

        vp0 = vanishingpts[0]
        vp0 /= vp0[2]

        vertical_line_inliers = compute_votes(edgelets, vp0, 3) > 0
        locations, directions, strengths, classes = edgelets

        edgelets = (
            locations[vertical_line_inliers], directions[vertical_line_inliers], strengths[vertical_line_inliers])
        locations, directions, strengths = edgelets

        vp_directions = locations - vp0[:2]

        angles = np.arctan2(vp_directions[:, 1], vp_directions[:, 0])

        s = np.argsort(np.sign(vp0[1]) * angles)
        angles = angles[s]
        locations = locations[s]
        directions = directions[s]
        strengths = strengths[s]

        planes_data = data["planes"]
        plane_parameters = np.array(data["planes"]["detection"][:, 6:9], dtype=np.float32)
        plane_offsets = np.linalg.norm(plane_parameters, axis=-1, keepdims=True)
        plane_normals = plane_parameters / np.maximum(plane_offsets, 1e-4)
        plane_clusters = np.array(data["planes"]["detection"][:, 4], dtype=np.int32)
        # roi = np.array(data["planes"]["detection"][:, 0:4], dtype=np.int32)
        cluster_prob = data["planes"]["detection"][:, 5]
        plane_masks = planes_data["masks"]
        number_planes = len(plane_masks)

        plane_XYZ = planes_data["plane_XYZ"][:, :, 80:-80, :].transpose(0, 2, 3, 1)

        plane_masks = resize_array(plane_masks, shape)
        plane_XYZ = resize_array(plane_XYZ, shape)

        # _log_ply(img_rs, [np.ones_like(plane_masks[0]),np.ones_like(plane_masks[0])], [np.float32(XYZ),XYZ_or], mult=2,
        #          file_path=logging_dir + '3D.ply')

        normals = cv2.resize(data["normals"], shape)
        logging_index = _log_image("normals.png", normals, logging_index=logging_index)

        normals = (normals - 127.5) / 127.5

        floor_indices, ceiling_indices, horiz_indices, floor_angs = find_floor_indices(floor_mask, ceiling_mask,
                                                                                       plane_masks, plane_normals)

        # print("floor indices are: ", floor_indices, horiz_indices)
        # print("ceiling indices are: ", ceiling_indices)

        floor_index = -1

        if len(floor_indices) > 0:
            floor_index = floor_indices[0]

        floor_normal = plane_normals[floor_index]
        # print("floor_normal", floor_normal)

        ceiling_index = -1

        if len(ceiling_indices) > 0:
            ceiling_index = ceiling_indices[0]

        wall_indices, vert_indices, vert_angs = find_wall_indices(wall_mask, plane_masks, plane_normals[floor_index],
                                                                  plane_normals)

        # print("wall indices are: ", wall_indices, vert_indices, vert_angs)

        basis_indices = np.int32(np.concatenate([floor_indices, ceiling_indices, wall_indices]))

        normals_combined = combined_normals(normals, plane_normals, plane_masks, basis_indices, cluster_prob)
        normals_c = normals_combined
        # normals_c = np.cross(floor_normal/2.-plane_normals[ceiling_index]/2., normals_combined)
        lengths = np.maximum(np.sqrt(np.sum(normals_c * normals_c, -1)), 1e-6)
        normals_c /= np.dstack((lengths, lengths, lengths))
        logging_index = _log_image("normals_c_org.png", 127.5 * (normals_c + 1), logging_index=logging_index)

        cluster_masks = [.03 * np.ones_like(plane_masks[0])]

        for i in range(1, 8):
            clust = np.nonzero(plane_clusters == i)[0]
            # clust = np.intersect1d(clust,vert_indices)

            if len(clust) > 0:
                cluster_masks.append(np.sum(plane_masks[clust], 0))

        plane_cluster_seg = np.argmax(cluster_masks, 0)
        logging_index = _log_segmentation_image("plane_cluster_seg.png", plane_cluster_seg, img_rs,
                                                logging_index=logging_index)

        st = time()
        other_markers = refine_surface(other_mask, l_image_rgb, big_thresh=.05, small_thresh=.95, watershed_dist=.05,
                                       gradient=False)
        wall_markers = np.int32(
            refine_surface(wall_mask, hed_rs, big_thresh=.05, small_thresh=.95, watershed_dist=.05, gradient=True,
                           watershed_mask=merged_lines == 0))
        floor_markers = np.int32(
            refine_surface(floor_mask, hed_rs, big_thresh=.05, small_thresh=.95, watershed_dist=.05, gradient=True,
                           watershed_mask=merged_lines == 0))
        wall_like_markers = np.int32(
            refine_surface(wall_like_mask, l_image_rgb, big_thresh=.05, small_thresh=.95, watershed_dist=.05,
                           gradient=False))
        ceiling_markers = np.int32(
            refine_surface(ceiling_mask, hed_rs, big_thresh=.05, small_thresh=.95, watershed_dist=.05, gradient=True,
                           watershed_mask=merged_lines == 0))

        # markers = join_segmentations(wall_like_markers,wall_markers)

        print("segmentation refined", time() - st)

        ceiling_prob = get_segmentation_image(ceiling_markers + 1, ceiling_mask, avg=True)
        wall_like_prob = get_segmentation_image(wall_like_markers + 1, wall_like_mask, avg=True)
        wall_prob = get_segmentation_image(wall_markers + 1, wall_mask, avg=True)
        floor_prob = get_segmentation_image(floor_markers + 1, floor_mask, avg=True)
        other_prob = get_segmentation_image(other_markers + 1, other_mask, avg=True)

        logging_index = _log_image("floor_markers.png", 255. * floor_prob, logging_index=logging_index)
        logging_index = _log_image("other_markers.png", 255. * other_prob, logging_index=logging_index)
        logging_index = _log_image("ceiling_markers.png", 255. * ceiling_prob, logging_index=logging_index)
        logging_index = _log_image("wall_like_markers.png", 255. * wall_like_prob, logging_index=logging_index)
        logging_index = _log_image("wall_markers.png", 255. * wall_prob, logging_index=logging_index)

        segmentation_initial = np.int32(np.argmax(np.dstack(
            (.25 * np.ones_like(other_mask), other_prob, floor_prob, wall_prob, ceiling_prob, wall_like_prob)), -1))
        segmentation_initial[merged_lines > 0] = 0
        segmentation_initial[np.logical_and(segmentation_initial == 3, wall_prob < .75)] = 6

        for i in range(1, 7):
            pruned = remove_small_objects(segmentation_initial == i, min_size=32)
            segmentation_initial[np.logical_and(segmentation_initial == i, pruned == 0)] = 0

        segmentation_initial = watershed(hed_rs, segmentation_initial,
                                         mask=Line.draw_all(line_data, np.ones_like(segmentation_initial), color=0,
                                                            thickness=1, sx=sx, sy=sy))

        logging_index = _log_segmentation_image("segmentation_refined.png", segmentation_initial, img_rs,
                                                logging_index=logging_index)

        sure_walls = segmentation_initial == 3

        # normals_c = cv2.resize(normals_c, (img.shape[1], img.shape[0]))
        normals_wall_like = normals_c.copy()

        labels_fan = np.int32(np.zeros((img_rs.shape[0], img_rs.shape[1])))
        fan_normals = []
        weights = []

        vp0[:2] *= [sx, sy]
        locations[:, 0] *= sx
        locations[:, 1] *= sy

        vl_image = np.zeros_like(all_lines)

        for i in range(-1, len(locations)):
            if i == -1:
                closest = np.int32([[0, img_rs.shape[1]], [0, 0]])
                dir = closest - vp0[:2]
                closest_index = np.argmin(np.abs(dir[:, 1]))
                pt1 = np.int32(2 * closest[closest_index] - vp0[:2])
                pt2 = np.int32(2 * locations[i + 1] - vp0[:2])
            elif i == len(locations) - 1:
                closest = np.int32([[img_rs.shape[1], img_rs.shape[0]], [img_rs.shape[1], 0]])
                dir = closest - vp0[:2]
                closest_index = np.argmin(np.abs(dir[:, 1]))
                pt2 = np.int32(2 * closest[closest_index] - vp0[:2])
                pt1 = np.int32(2 * locations[i] - vp0[:2])
            else:
                pt1 = np.int32(2 * locations[i] - vp0[:2])
                pt2 = np.int32(2 * locations[i + 1] - vp0[:2])

            triangle = np.array([[[vp0[0], vp0[1]], pt1, pt2]], np.int32)

            arc_mask = cv2.fillPoly(np.zeros_like(labels_fan), pts=triangle, color=1)

            wall_arc = np.logical_and(sure_walls > 0, arc_mask > 0)
            a = np.sum(wall_arc)

            if a > 0:
                labels_fan[wall_arc > 0] = i + 2
                cur_normal = np.mean(normals_c[np.logical_and(wall_arc, wall_mask > .9)], 0)
                cur_normal /= max(np.linalg.norm(cur_normal), .00001)
                fan_normals.append(cur_normal)
                weights.append(a)

        # Merge
        k = 1
        ang_threshold = .8
        for i in range(len(fan_normals) - 1):
            cur_normal = fan_normals[i]
            next_normal = fan_normals[i + 1]

            ang1 = abs(np.dot(cur_normal, next_normal))

            if ang1 > ang_threshold:
                labels_fan[labels_fan == i + 1] = k
                labels_fan[labels_fan == i + 2] = k
                # fan_normals[i+1] = (weights[i]*cur_normal+ weights[i+1]*next_normal)/(weights[i]+weights[i+1])
                norm = np.mean(normals_c[np.logical_and(labels_fan == k, wall_mask > .9)], 0)
                norm /= max(np.linalg.norm(norm), .00001)
                fan_normals[i] = norm
                fan_normals[i + 1] = norm
                normals_wall_like[labels_fan == k] = norm

            else:
                k += 1
                # labels_fan[mask > 0] = k
                # normals_wall_like[mask] = next_normal
                # cv2.polylines(vl_image, pts=np.array([[triangle[0,0], triangle[0,2]]]), isClosed=True, thickness=1, color=1)

            # pre_normal = np.zeros_like(floor_normal)
            # if i == -1 or i == len(locations):
            #     pre_normal = np.mean(normals_c[wall_arc], 0)
            #     pre_normal /= max(np.linalg.norm(pre_normal), .00001)
            #     labels_fan[arc_mask > 0] = k
            #
            #     cv2.polylines(vl_image, pts=np.array([[triangle[0,0], triangle[0,2]]]), isClosed=True, thickness=2, color=1)
            # else:
            #     pre_normal = np.mean(normals_c[np.logical_and(sure_walls>0, labels_fan == k)], 0)
            #     pre_normal /= max(np.linalg.norm(pre_normal), .00001)
            #
            # cur_normal = np.mean(normals_c[wall_arc], 0)
            # cur_normal /= max(np.linalg.norm(cur_normal), .00001)
            #
            # ang = abs(np.dot(cur_normal, pre_normal))
            #
            # if ang > ang_threshold:
            #     labels_fan[arc_mask > 0] = k
            #     normals_wall_like[np.logical_and(sure_walls>0, labels_fan == k)] = pre_normal
            #
            # else:
            #     fan_normals.append(cur_normal)
            #     k += 1
            #     labels_fan[arc_mask > 0] = k
            #     cv2.polylines(vl_image, pts=np.array([[triangle[0,0], triangle[0,2]]]), isClosed=True, thickness=1, color=1)

        labels_fan[sure_walls == 0] = 0

        vl_image[sure_walls == 0] = 0
        logging_index = _log_segmentation_image("fan.png", labels_fan, img_rs, logging_index=logging_index)
        logging_index = _log_image("normals_wall_org.png", 127.5 * (normals_wall_like + 1), logging_index=logging_index)
        logging_index = _log_segmentation_image("vl_image.png", vl_image, img_rs, logging_index=logging_index)
        # stop

        plane_classes = get_planes_class(plane_masks, ade_seg)
        # print("plane_classes", plane_classes)

        full_planes = plane_masks.copy()
        full_planes[full_planes < .01] = 0

        plane_masks[plane_masks < .5] = 0
        wall_like_planes = [np.zeros_like(hed_rs / 255.)]

        wall_like_indices = []
        # wall_indices = []
        floor_indices = []
        ceiling_indices = []

        for k in range(number_planes):
            a = np.int32(plane_classes[k])

            name = "plane_" + str(k)
            # for a in pc[0]:
            name += "_" + str(a)

            # logging_index = _log_image(name + ".png", 255. * plane_masks[k])
            if a == 3 or a == 5:
                if k not in wall_like_indices:
                    wall_like_planes.append(plane_masks[k])
                    wall_like_indices.append(k)
                continue

            if a == 2:
                floor_indices.append(k)
                continue

            if a == 4:
                ceiling_indices.append(k)
                continue

        wall_like_planes = np.float32(wall_like_planes)
        wall_planes_seg = np.argmax(wall_like_planes, 0)

        for i in np.unique(wall_planes_seg):
            if i == 0: continue
            mask = skeletonize(wall_planes_seg == i)
            wall_planes_seg[np.logical_and(mask == 0, wall_planes_seg == i)] = 0

        logging_index = _log_segmentation_image("wall_planes_seg.png", wall_planes_seg, img_rs,
                                                logging_index=logging_index)

        labels_wall_1 = labels_fan
        labels_wall_1[labels_wall_1 > 0] += np.amax(wall_planes_seg) + 1
        logging_index = _log_segmentation_image("labels_wall_graph.png", labels_fan, img_rs, avg=False,
                                                logging_index=logging_index)

        for i in np.unique(labels_wall_1):
            if i == 0: continue

            mask = i == labels_wall_1
            m = mode(wall_planes_seg[np.logical_and(wall_planes_seg > 0, mask)])

            if len(m[0]) > 1:
                labels_wall_1[mask] = m[0]

        logging_index = _log_segmentation_image("labels_wall_merge1.png", labels_wall_1, img_rs, avg=False,
                                                logging_index=logging_index)

        label_indices = np.unique(labels_wall_1)

        # normals_labels_1 = get_segmentation_image(labels_wall_1, normals_c, avg=True)
        # logging_index = _log_image("normals_labels_1.png", 127.5 * (normals_labels_1 + 1), logging_index=logging_index)

        wall_planes_number = len(wall_like_indices)

        means = np.zeros((len(label_indices), wall_planes_number + 1), dtype=np.float32)

        for i in range(1, wall_planes_number + 1):
            means[:, i] = ndimage.mean(full_planes[wall_like_indices[i - 1]], labels=labels_wall_1, index=label_indices)

        means[means < .03] = 0
        arg = np.argmax(means, axis=-1)

        labels_arg = np.zeros_like(np.int32(labels_wall_1))

        for i in range(1, len(label_indices)):
            if arg[i] > 0:
                labels_arg[labels_wall_1 == label_indices[i]] = wall_like_indices[arg[i] - 1] + 1

            else:
                # print("if no good match just take the closest by angle")
                normal = np.mean(normals_c[labels_wall_1 == label_indices[i]], 0)
                normal /= max(np.linalg.norm(normal), .00001)
                wall_index = np.argmax(np.abs(np.dot(plane_normals[vert_indices], normal)))
                labels_arg[labels_wall_1 == label_indices[i]] = vert_indices[wall_index] + 1

        # logging_index = _log_image('labels_arg.png', get_segmentation_image(np.int32(labels_arg), img_rs, avg=False), logging_index=logging_index)

        # needs to be cleaned up
        labels_wall_1 = labels_arg.copy()
        labels_wall_1[labels_wall_1 > 0] += np.amax(wall_planes_seg)
        labels_wall_2 = labels_wall_1.copy()

        for i in np.unique(labels_wall_1):
            if i == 0: continue

            label_mask = labels_wall_1 == i

            intersection = np.sum(label_mask[wall_planes_seg > 0])

            if intersection > 1:
                w = watershed(hed_rs, wall_planes_seg, mask=label_mask)
                l = np.logical_and(w > 0, label_mask > 0)
                labels_wall_2[l] = w[l]

        logging_index = _log_segmentation_image("labels_wall_merge.png", labels_wall_2, img_rs, avg=False,
                                                logging_index=logging_index)

        final_masks = []

        final_plane_parameters = []
        final_plane_XYZ = []

        wall_areas = []
        for l in np.unique(labels_wall_2):
            if l == 0: continue

            mask = np.uint8(labels_wall_2 == l)
            area = np.sum(mask)

            if area > 16 * 12:
                wall_areas.append((area))
                final_masks.append(255 * mask)

                l = int(np.median(labels_arg[mask > 0])) - 1

                plane_parameter = np.zeros(10)
                plane_parameter[:9] = data["planes"]["detection"][l][:9]
                plane_parameter[6:9] = plane_parameters[l]
                plane_parameter[9] = 2

                final_plane_parameters.append(plane_parameter)
                final_plane_XYZ.append(plane_XYZ[l])

        # sort wall planes, largest to smallest/ugly due to sheer laziness
        area_sort = np.argsort(wall_areas)[::-1]
        final_masks = np.array(final_masks)[area_sort].tolist()
        final_plane_parameters = np.array(final_plane_parameters)[area_sort].tolist()
        final_plane_XYZ = np.array(final_plane_XYZ)[area_sort].tolist()

        # add floor
        if len(floor_indices) > 0:
            floor_mask = np.uint8(segmentation_initial == 2)
            # floor_mask = cv2.dilate(floor_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            final_masks.append(255 * floor_mask)

            plane_parameter = np.zeros((10))
            plane_parameter[:9] = data["planes"]["detection"][floor_indices[0]][:9]
            plane_parameter[6:9] = plane_parameters[floor_indices[0]]
            plane_parameter[9] = 1

            final_plane_parameters.append(plane_parameter)
            final_plane_XYZ.append(plane_XYZ[floor_indices[0]])

        # add ceiling
        if len(ceiling_indices) > 0:
            ceiling_mask = 255 * np.uint8(segmentation_initial == 4)
            # ceiling_mask = cv2.dilate(ceiling_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            # ceiling_mask[segmentation_initial != 1] = 0
            final_masks.append(ceiling_mask)
            plane_parameter = np.zeros((10))
            plane_parameter[:9] = data["planes"]["detection"][ceiling_indices[0]][:9]
            plane_parameter[6:9] = plane_parameters[ceiling_indices[0]]
            plane_parameter[9] = 3
            final_plane_parameters.append(plane_parameter)
            final_plane_XYZ.append(plane_XYZ[ceiling_indices[0]])

        final_plane_parameters = np.float32(final_plane_parameters)

        final_plane_number = len(final_masks)

        # print("final_plane_number", final_plane_number)
        final_masks = np.uint8(final_masks)

        final_labels = np.argmax(final_masks, 0)

        final_labels[final_labels > 0] += 1
        final_labels[final_masks[0] > 0] = 1
        final_labels += 1
        final_labels = np.uint8(final_labels)

        length_threshold = 32
        canny_aperture_size = 7

        fld = cv2.ximgproc.createFastLineDetector(_length_threshold=length_threshold,
                                                  _canny_aperture_size=canny_aperture_size)
        final_labels[vl_image > 0] = 0
        lines = fld.detect(final_labels)

        final_line_data = []
        for line in line_data:
            # t = final_labels_rs[int(line.midpoint[1]),int(line.midpoint[0])]
            # if t==0:
            final_line_data.append(line)

        if lines is not None:
            for line in lines:
                new_line = Line(line[0][0] / sx, line[0][1] / sy, line[0][2] / sx, line[0][3] / sy)
                if new_line.get_confidence() > 0.1: final_line_data.append(new_line)

        final_line_data = Line.merge(final_line_data, 3, search_length=1.01, angle_threshold=math.radians(1.0))

        final_masks_rs = resize_array(np.uint8(final_masks), (img.shape[1], img.shape[0]))
        final_labels_rs = np.int32(np.argmax(final_masks_rs, 0))

        final_labels_rs[final_labels_rs > 0] += 1
        final_labels_rs[final_masks_rs[0] > 0] = 1
        final_labels_rs += 1

        final_merged_lines = Line.draw_all(final_line_data, np.zeros_like(final_labels_rs), color=1, thickness=1)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for i in np.unique(final_labels_rs):
            mask = final_labels_rs == i
            inter = np.logical_and(mask, final_merged_lines)
            final_labels_rs[mask > 0] = 0
            mask[final_merged_lines > 0] = 0
            pruned = remove_small_objects(mask, 16)
            pruned[inter > 0] = 1
            filled = np.uint8(remove_small_holes(pruned, 16))

            final_labels_rs[filled > 0] = i

        final_labels_rs = np.uint8(final_labels_rs)
        final_labels_rs[final_merged_lines > 0] = 0
        logging_index = _log_segmentation_image("pre_final_labels.png", final_labels_rs, img,
                                                logging_index=logging_index)

        # for i in range(1, final_plane_number + 2):
        #     contours, hierarchy = cv2.findContours(np.uint8(final_labels_rs == i), cv2.RETR_TREE,
        #                                            cv2.CHAIN_APPROX_SIMPLE)
        #     if len(contours) > 0:
        #         for i in range(len(contours)):
        #             cv2.drawContours(final_labels_rs, [contours[i]], -1, 0, 3, lineType=cv2.LINE_8)

        # final_labels_rs = cv2.threshold(np.uint8(final_labels_rs),0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[0]

        new_img = img.copy()
        new_img[final_merged_lines > 0] = (255, 255, 255)

        logging_index = _log_image("img_lines.png", new_img, logging_index=logging_index)

        final_labels = np.int32(final_labels_rs)

        final_labels = cv2.watershed(new_img, final_labels)
        final_labels[final_labels < 0] = 0
        logging_index = _log_segmentation_image("final_labels.png", np.int32(final_labels), img,
                                                logging_index=logging_index)

        data["planes"]["masks"] = np.zeros((final_plane_number, img.shape[0], img.shape[1]), dtype=np.uint8)

        data["planes"]["detection"] = np.zeros((final_plane_number, 10), dtype=data["planes"]["detection"].dtype)
        data["planes"]["detection"] = final_plane_parameters

        mask_contours = []

        for d in range(final_plane_number):
            contours, hierarchy = cv2.findContours(np.uint8(final_labels == d + 2), cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            data["planes"]["masks"][d] = np.zeros_like(np.uint8(final_labels == d + 2))

            plane_contours = []

            if len(contours) > 0:

                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])

                    if area > 16 * 16:
                        if hierarchy[0, i, 3] == -1:  # this is the outer contour which we need to draw
                            cv2.drawContours(data["planes"]["masks"][d], [contours[i]], -1, 255, -1)

                            # cv2.drawContours(img_rs, [contours[i]], -1,(0,0,255), 1, lineType=cv2.LINE_AA)
                            # epsilon = .25*cv2.arcLength(contours[i], True) / max(shape[0],shape[1])
                            # contours[i] = cv2.approxPolyDP(contours[i], epsilon, closed=True)
                            plane_contours.append(contours[i].tolist())
                        else:
                            cv2.drawContours(data["planes"]["masks"][d], contours, i, 0, -1)

                rect = cv2.boundingRect(data["planes"]["masks"][d])

                data["planes"]["detection"][d, 0:4] = [
                    rect[1], rect[0], rect[1] + rect[3], rect[0] + rect[2]
                ]

            mask_contours.append(plane_contours)

            data["planes"]["contours"] = mask_contours

        print("time", time() - org_time)
        _log_ply(img_rs, np.uint8(data["planes"]["masks"]), np.float32(final_plane_XYZ), mult=2,
                 file_path=logging_dir + '3D_refine.ply')
