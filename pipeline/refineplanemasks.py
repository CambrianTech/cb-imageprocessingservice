from pipeline.core import PipelineStep
import cv2
import numpy as np
from scipy import ndimage
from time import time

import cambrian.image_processing as ip
from cambrian import geometry
from cambrian.frei_chen import frei_chen
from cambrian.Line import Line
from cambrian.transformations import euler_from_matrix
from skimage.segmentation import join_segmentations
from skimage.morphology import skeletonize

import pickle
import math
from skimage.segmentation import watershed
from scipy.stats import mode

from skimage.morphology import remove_small_objects, remove_small_holes

IM_LOGGING_ENABLED = False
IM_LOGGING3D_ENABLED = False

furniture_labels = [15, 30, 23, 64, 97, 44, 35,19, 7, 69, 75, 93, 110]
wall_like = [0, 8, 14, 18, 22, 24, 42, 58,63, 130]
wall_int = [3, 8, 22, 100]
f=1.0
METADATA = np.array([571.87, 571.87, 320, 240, 640, 480, 0, 0, 0, 0])

IMAGE_MAX_DIM = 640
IMAGE_MIN_DIM = 480


def _log_image(logging_dir, name, image, logging_index=0, logging=IM_LOGGING_ENABLED):
    if logging:
        name = logging_dir + str(logging_index) + " - " + name
        print("name", name)
        cv2.imwrite(name, image)
    return logging_index + 1

def _log_segmentation_image(logging_dir, name, segmentation, image, avg=False, logging_index=0):
    if IM_LOGGING_ENABLED:
        name = logging_dir + str(logging_index) + " - " + name
        seg = get_segmentation_image(segmentation, image, avg)
        cv2.imwrite(name, seg)
    return logging_index + 1

def _log_ply(logging_dir, name, image, masks, plane_XYZ, write_occlusion=False, mult=1.0, logging_index = 0):
    if IM_LOGGING3D_ENABLED:
        file_path = logging_dir + str(logging_index) + " - " + name
        image = cv2.resize(image, (int(mult * 160), int(mult * 120)))
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
            indices = np.nonzero(np.logical_and(segmentation == mask_index, masks[mask_index] > 0.01))
            # indices = np.nonzero(masks[mask_index] > .05)
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
    return logging_index + 1


def rough_dilate_erode(is_dilate, mask, size=5, iterations=1, scale=0.5, maintain_size=True, interpolation=cv2.INTER_NEAREST):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    shape = mask.shape
    mask = cv2.resize(mask, (int(shape[1] * scale), int(shape[0] * scale)), interpolation)
    mask = cv2.dilate(mask, kernel, iterations=iterations) if is_dilate else cv2.erode(mask, kernel, iterations=iterations)
    if maintain_size:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation)
    return mask


def find_lines(img, gradient, normals):

    height, width = img.shape[:2]
    diagonal = np.hypot(width, height)
    # print("image w,h", width, height)

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

    lines_c = None

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
            l = new_line.length
            # mp = np.int32(new_line.midpoint)
            # pt1 = np.int32(new_line.point_a)
            # pt2 = np.int32(new_line.point_b)
            cm = new_line.get_color_mean()
            # print("color_mean", cm)
            if cm[3]<2 or l < int(diagonal / 60.0): continue

            # if mask[mp[1],mp[0]]>0: continue
            conf = new_line.get_confidence()
            if conf > 0.2:
                confs.append(conf)
                line_data.append(new_line)

    line_data = [line_data[i] for i in np.argsort(confs)]

    line_data = Line.merge(line_data, diagonal / 120.0, search_length=1.03, angle_threshold=math.radians(3.0))

    # # # #
    line_data, intersections = Line.find_corners(line_data, search_length=1.5, angle_threshold=math.radians(5.0),
                                                 parallel_threshold=math.radians(5), confidence_diff=0.4)

    return line_data, lines



def calcPlaneXYZ(planes, width=IMAGE_MIN_DIM, height=IMAGE_MAX_DIM, camera=METADATA, max_depth=10):
    urange = (np.arange(width, dtype=np.float32) / (width) * (camera[4]) - camera[2]) / camera[0]
    urange = urange.reshape(1, -1).repeat(height, 0)

    vrange = (np.arange(height, dtype=np.float32) / (height) * (camera[5]) - camera[3]) / camera[1]
    vrange = vrange.reshape(-1, 1).repeat(width, 1)

    ranges = np.stack([urange, np.ones(urange.shape), -vrange], axis=-1)

    planeOffsets = np.linalg.norm(planes, axis=-1, keepdims=True)
    planeNormals = planes / np.maximum(planeOffsets, 1e-4)

    normalXYZ = np.dot(ranges, planeNormals.transpose())
    # normalXYZ = np.round(normalXYZ,2)


    normalXYZ[normalXYZ == 0] = 1e-4

    planeDepths = planeOffsets.squeeze(-1) / normalXYZ
    if max_depth > 0:
        planeDepths = np.clip(planeDepths, 0, max_depth)
        pass
    XYZ = (np.expand_dims(planeDepths, -1) * np.expand_dims(ranges, 2))
    # cv2.imwrite("n.png", cv2.normalize(XYZ.transpose(2, 0, 1, 3)[0], None, 0, 255, cv2.NORM_MINMAX))
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

def xyz_to_uv(pt, width=IMAGE_MAX_DIM, height=IMAGE_MIN_DIM, camera=METADATA, f=1.0):
    pt = pt / pt[1]

    u = (pt[0] * camera[0]*f + camera[2]) * width / camera[4]
    v = (-pt[2] * camera[1]*f + camera[3]) * height / camera[5]

    return np.array([u, v])


def uv_to_xyz(uv, depth, width=IMAGE_MAX_DIM, height=IMAGE_MIN_DIM, camera=METADATA, f=1.0):
    p0 = (uv[0] * camera[4] / width - camera[2]) / (f*camera[0]) * depth
    p2 = - (uv[1] * camera[5] / height - camera[3]) / (f*camera[1]) * depth

    return np.array([p0, depth, p2])

def proj(points, plane):
    plane_offset = np.linalg.norm(plane)
    plane_normal = plane / plane_offset
    s = np.dot(points, plane_normal) - plane_offset
    return s


#######################################

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


##############################################################
# Vanishing pt functions

def compute_edgelets(lines):
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
    # classes = []

    # check_class = False
    # if len(class_labels)>0:
    #     check_class = True

    for l in lines:
        l = l[0]
        p0, p1 = np.array([l[0], l[1]]), np.array([l[2], l[3]])
        c = (p0 + p1) / 2
        # if check_class:
        #     classes.append(class_labels[int(c[1]), int(c[0])])

        locations.append(c)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)
    # classes = np.array(classes)

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

def remove_inliers(model, edgelets, threshold_inlier=10):
    """Remove all inlier edglets of a given model.
    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.
    Returns
    -------
    edgelets_new: tuple of ndarrays
        All Edgelets except those which are inliers to model.
    """
    inliers = compute_votes(edgelets, model, threshold_inlier) > 0
    locations, directions, strengths = edgelets
    locations = locations[~inliers]
    directions = directions[~inliers]
    strengths = strengths[~inliers]
    # classes = classes[~inliers]
    edgelets = (locations, directions, strengths)
    return edgelets

def ransac_vanishing_point(edgelets, lines, num_ransac_iter=2000, threshold_inlier=5, max_time=10.0, line_indices=None, roi=None
                           , vp=None,camera=None, up_normal=None):
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

    if line_indices is not None:
        edgelets = locations[line_indices], directions[line_indices], strengths[line_indices]
        locations, directions, strengths = edgelets
        lines = lines[line_indices]

    num_pts = strengths.size
    arg_sort = np.argsort(-strengths)

    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_models = None
    best_votes = 0
    model_inliers = None

    t = time()

    for ransac_iter in range(num_ransac_iter):
        if time() - t > max_time or  len(first_index_space)==0 or len(second_index_space)==0:
            return best_models, best_votes, model_inliers
        # print(len(first_index_space), first_index_space)
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        if ind1==ind2: continue

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
            # bad
            continue

        angle = 0
        axis_up = [0,0,0]
        if vp is not None:
            K=camera
            axis_up = geometry.unit_vector(np.cross(np.dot(np.linalg.inv(K), vp), np.dot(np.linalg.inv(K), current_model)))
            angle = np.dot(up_normal, -np.sign(axis_up[2])*axis_up)
            if np.degrees(np.arccos(angle))>10:
                continue


        good = compute_votes(
            edgelets, current_model, threshold_inlier)

        current_votes = (good*strengths).sum()

        if current_votes> best_votes:
            best_models = current_model
            best_votes = current_votes
            model_inliers = good
            # print(np.degrees(np.arccos(angle)), up_normal,-np.sign(-axis_up[2])*axis_up)
        continue

    inlier_indices = np.nonzero(model_inliers)[0]

    if line_indices is not None:
        inlier_indices = line_indices[inlier_indices]

    if best_models is not None:
        print("ransac 2 line", np.int32(best_models / best_models[2]))
    return best_models, best_votes, inlier_indices




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

    plane_normals_nn = plane_normals.copy()
    number_planes = len(plane_normals)
    cluster_indices = cluster_prob > .5
    # print("good clusters", cluster_indices)
    basis_indices = basis_indices[cluster_indices[basis_indices]]

    for i in range(number_planes):
        if cluster_indices[i]:
            plane_normals_nn[i] = mode(normals[plane_masks[i] > np.amax(plane_masks[i]) / 2.], axis=0)[0]
            plane_normals_nn[i] /= np.linalg.norm(plane_normals_nn[i])


    # plane_normals_nn[0] = -plane_normals_nn[0]

    R, _ = calcTransformation(plane_normals_nn[basis_indices], plane_normals[basis_indices])
    # R = np.eye(3)

    nr = normals.reshape((-1, 3))
    normals = np.matmul(R, nr.transpose()).transpose().reshape(normals.shape)

    lengths = np.maximum(np.sqrt(np.sum(normals * normals, -1)), 1e-6)
    normals /= np.dstack((lengths, lengths, lengths))

    plane_normals_nn[basis_indices] = np.matmul(R,plane_normals_nn[basis_indices].transpose()).transpose()


    for i in range(number_planes):
        # if cluster_indices[i]:
        m = plane_masks[i].copy()
        # m[m>.5] = 1
        mult = np.dstack((m,m,m))

        normals = (1.0 - mult) * normals + mult * plane_normals[i]

    lengths = np.maximum(np.sqrt(np.sum(normals * normals, -1)), 1e-6)
    normals /= np.dstack((lengths, lengths, lengths))
    return normals, plane_normals_nn


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

def merge_by_angle_sweep(labels_fan, normals_img, fan_normals, mask, angle_threshold):

    k=1
    normals_wall = normals_img.copy()
    fan_normals_reduced = []

    for i in range(len(fan_normals) - 1):

        cur_normal = fan_normals[i]
        next_normal = fan_normals[i + 1]
        ang1 = abs(np.dot(cur_normal, next_normal))
        labels_fan[labels_fan == i + 1] = k

        if ang1 > angle_threshold:
            # print('angle threshold met', ang_threshold, ang1)

            labels_fan[labels_fan == i + 2] = k
            wedge = np.logical_and(labels_fan == k, mask)
            norm = np.mean(normals_img[wedge], 0)

            if ~np.isnan(norm[0]):
                norm /= max(np.linalg.norm(norm), .00001)

                fan_normals[i] = norm
                fan_normals[i + 1] = norm

                normals_wall[labels_fan == k] = norm

                if i==len(fan_normals) - 2:
                    fan_normals_reduced.append(norm)
                    k += 1
            else:
                normals_wall[labels_fan == k] = cur_normal

                if i == len(fan_normals) - 2:
                    fan_normals_reduced.append(norm)
                    k += 1
        else:
            # print('angle threshold not met', angle_threshold, ang1)
            wedge = np.logical_and(labels_fan == k, mask)
            norm = np.mean(normals_img[wedge], 0)

            if ~np.isnan(norm[0]):
                norm /= max(np.linalg.norm(norm), .00001)
                fan_normals_reduced.append(norm)
            else:
                fan_normals_reduced.append(norm)
            k += 1

    fan_normals_reduced = np.float32(fan_normals_reduced)

    return labels_fan, fan_normals_reduced, normals_wall


def resize_array(array, shape):
    length = len(array)

    dim=1
    array_lr = np.zeros((length, shape[1], shape[0]))

    if array[0].ndim>2:
        dim = array.shape[-1]
        array_lr = np.zeros((length, shape[1], shape[0], dim))

    for k in range(length):
        array_lr[k] = cv2.resize(array[k], shape)

    return array_lr


def camera_fov_res_to_intrinsics(fov: float, res: np.ndarray):
    # https://stackoverflow.com/a/41137160
    # fov = 2 * arctan(r / (2 * f)) <=> f_y = r / (2 * tan(fov / 2))
    c = res / 2

    # Assume the fov corresponds to the longest side and use that for focal
    i = 1 if c[0] >= c[1] else 0
    f = c[i] / np.tan(np.radians(fov) / 2)
    K = np.array([f, f, c[0], c[1], res[0], res[1]], dtype=np.float32)

    return K

def estimate_fov(v1, v2, width = IMAGE_MAX_DIM, height = IMAGE_MIN_DIM, camera = METADATA):
    # https://stackoverflow.com/a/41137160
    # fov = 2 * arctan(r / (2 * f)) <=> f_y = r / (2 * tan(fov / 2))

    e1 = [(v1[0] * camera[4] / width - camera[2]) / (width/2), (v1[1] * camera[5] / height - camera[3]) /(height/2)]
    e2 = [(v2[0] * camera[4] / width - camera[2]) / (width/2), (v2[1] * camera[5] / height - camera[3]) / (height/2)]
    e1 = geometry.unit_vector(e1)
    e2 = geometry.unit_vector(e2)
    f = np.sqrt(np.dot(e1,e2))
    # f = f*np.float32((width/2,height/2))
    fov = 2* np.arctan(1 / f )
    print("e1,e2", e1,e2,f,np.degrees(fov))
    return np.degrees(fov), f

################################## not used
# def get_XYZ_from_depth(depth, width=IMAGE_MIN_DIM, height=IMAGE_MAX_DIM, camera=METADATA, max_depth=10):
#     height = depth.shape[0]
#     width = depth.shape[1]
#
#     urange = (np.arange(width, dtype=np.float32) / (width) * (camera[4]) - camera[2]) / camera[0]
#     urange = urange.reshape(1, -1).repeat(height, 0)
#
#     vrange = (np.arange(height, dtype=np.float32) / (height) * (camera[5]) - camera[3]) / camera[1]
#     vrange = vrange.reshape(-1, 1).repeat(width, 1)
#
#     X = depth * urange
#     Y = depth
#     Z = -depth * vrange
#     XYZ = np.dstack((X,Y,Z))
#
#     return XYZ

#
#
def draw_grid(img, line_color=(0, 255, 0), thickness=1, type_=cv2.LINE_AA, pxstep=50):
    '''(ndarray, 3-tuple, int, int) -> void
    draw gridlines on img
    line_color:
        BGR representation of colour
    thickness:
        line thickness
    type:
        8, 4 or cv2.LINE_AA
    pxstep:
        grid line frequency in pixels
    '''
    x = pxstep
    y = pxstep
    while x < img.shape[1]:
        cv2.line(img, (x, 0), (x, img.shape[0]), color=line_color, lineType=type_, thickness=thickness)
        x += pxstep

    while y < img.shape[0]:
        cv2.line(img, (0, y), (img.shape[1], y), color=line_color, lineType=type_, thickness=thickness)
        y += pxstep
    return img

def rotationMatrixToEulerAngles(R):

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z])


def fov_to_focal(fov, length):
    return length / (2 * np.tan(np.radians(fov) / 2))

def focal_to_fov(focal_length, length):
    return 2 * np.arctan2(length,(2 * focal_length))

def camera_fov_res_to_intrinsics(fov: float, res: np.ndarray):

    c = res / 2
    f = c[0] / np.tan(np.radians(fov) / 2)
    K = np.array([f, f, c[0], c[1], res[0], res[1]], dtype=np.float32)

    return K, f

def camera_fov_to_intrinsic_matrix(fov,w,h):
    K = np.eye(3)

    K[0, :] = [fov_to_focal(fov, w),  w/2,0]
    K[1, :] = [0, h/2, -fov_to_focal(fov, w)]
    K[2, :] = [0, 1, 0]
    return K

def camera_focal_length_to_intrinsic_matrix(focal_length, w, h):
    K = np.eye(3)

    K[0, :] = [focal_length,  w/2,0]
    K[1, :] = [0, h/2, -focal_length]
    K[2, :] = [0, 1, 0]
    return K

def get_edgelets_close_to_dir(edgelets, direction, threshold):

    dots = abs(edgelets[1] * direction)
    close_edgelets = np.nonzero(dots>1.0-threshold)[0]
    return close_edgelets


def get_edgelets_pointing_to_point(edgelets, point,threshold):

    locations, directions, _ = edgelets

    desiredDir = locations - point
    dir_norm = np.linalg.norm(desiredDir, axis=1)
    dir_norm[dir_norm == 0] = 1e-5

    dots = abs(np.sum(desiredDir*directions, axis=1))/dir_norm
    close_edgelets = np.nonzero(dots > 1.0 - threshold)[0]
    return close_edgelets

def compute_normal_from_vps(edgelets,img, fov, floor_normal, floor_offset, floor_mask):
    e_lines = edgelet_lines(edgelets)

    pp = [img.shape[1]/2, img.shape[0]/2]
    vps=[]
    inliers = []

    vertical_edgelet_indices = get_edgelets_close_to_dir(edgelets,[0,1],.03)
    vp_vertical, votes, inliers_vertical = ransac_vanishing_point(edgelets, e_lines, 2000, threshold_inlier=1, max_time=1.0, line_indices=vertical_edgelet_indices)

    if vp_vertical is not None:
        vps.append(vp_vertical)
        inliers.append(inliers_vertical)

    horizontal1_edgelet_indices = get_edgelets_close_to_dir(edgelets, [1, 0], .5)

    vp_horizontal1, votes, inliers_horizontal1 = ransac_vanishing_point(edgelets, e_lines, 2000, threshold_inlier=1, max_time=1.0, line_indices=horizontal1_edgelet_indices)

    if vp_horizontal1 is not None:
        vps.append(vp_horizontal1)
        inliers.append(inliers_horizontal1)

    horizontal2_edgelet_indices = get_edgelets_close_to_dir(edgelets, [1, 0], .95)
    horizontal2_edgelet_indices = np.setdiff1d(horizontal2_edgelet_indices, np.nonzero(compute_votes(edgelets,vp_horizontal1,10))[0])
    horizontal2_edgelet_indices = np.setdiff1d( horizontal2_edgelet_indices, np.nonzero(compute_votes(edgelets,vp_vertical,10))[0])

    vp_horizontal2, votes, inliers_horizontal2 = ransac_vanishing_point(edgelets, e_lines, 2000, threshold_inlier=2,
                                                                      max_time=1.0,
                                                                      line_indices=horizontal2_edgelet_indices)

    if vp_horizontal2 is not None:
        vps.append(vp_horizontal2)
        inliers.append(inliers_horizontal2)

    horizontal2_edgelet_indices = np.setdiff1d(horizontal2_edgelet_indices,
                                               np.nonzero(compute_votes(edgelets, vp_horizontal2,5))[0])

    vp_horizontal3, votes, inliers_horizontal3 = ransac_vanishing_point(edgelets, e_lines, 2000, threshold_inlier=1,
                                                                        max_time=1.0,
                                                                        line_indices=horizontal2_edgelet_indices)
    if vp_horizontal3 is not None:
        vps.append(vp_horizontal3)
        inliers.append(inliers_horizontal3)

    vps = np.float32(vps)

    axes = [None, None, None]
    axes_indices = []
    max_det = .9

    centered = (vps[:,:2]/np.dstack((vps[:,2],vps[:,2]))-pp)/pp
    forward_indices = np.logical_and((centered[0,:,0]<2.0), (centered[0,:,1]<2.0))

    index1 = [True,True,True,True]
    if np.sum(forward_indices)>0:
        index1 = forward_indices

    for i in range(len(vps)):
        for j in range(len(vps)):
            if i>=j: continue
            for k in range(len(vps)):
                if j>=k: continue
                s = index1[i] + index1[j] + index1[k]
                if s==0: continue

                vp0 = vps[i]
                vp1 = vps[j]
                vp2 = vps[k]
                fov_pos = [50,70,90]

                for f in fov_pos:
                    K = camera_fov_to_intrinsic_matrix(f, w=img.shape[1], h=img.shape[0])
                    K_inv = np.linalg.inv(K)

                    new_vps = [vp0, vp1, vp2]
                    axes_new = np.dot(K_inv, np.transpose(new_vps)).transpose()
                    lengths = np.linalg.norm(axes_new, axis=-1)
                    axes_new = np.float32(axes_new / np.dstack((lengths, lengths, lengths)))[0]
                    coors = np.argmax(abs(axes_new), 0)
                    axes_new = axes_new[coors]
                    DET = abs(np.linalg.det(axes_new))

                    if DET > max_det:

                        max_det = DET
                        axes_indices = np.int32([i,j,k])
                        axes_indices = axes_indices[coors]
                        axes = np.zeros_like(axes_new)
                        axes = axes_new

                        axes[2] = geometry.unit_vector(np.cross(axes[0],axes[1]))
                        fov=f


    if axes[2] is not None:
        j, k,_ = axes_indices
        vp1 = vps[j]
        vp2 = vps[k]
        v1 = vp1[:2] / vp1[2] - pp
        v2 = vp2[:2] / vp2[2] - pp
        p_f = -v1[0] * v2[0] - v1[1] * v2[1]

        if p_f > 0:
            focal_length = np.sqrt(p_f)
            fov_new = np.degrees(focal_to_fov(focal_length, 2 * pp[0]))
            if fov_new>40 and fov_new<110:
                fov=fov_new
                K = camera_fov_to_intrinsic_matrix(fov, w=img.shape[1], h=img.shape[0])
                K_inv = np.linalg.inv(K)

                new_vps = [vp1, vp2]
                axes_new = np.dot(K_inv, np.transpose(new_vps)).transpose()
                lengths = np.linalg.norm(axes_new, axis=-1)
                axes_new = np.float32(axes_new / np.dstack((lengths, lengths, lengths)))[0]
                axes = np.float32([axes_new[0], axes_new[1], geometry.unit_vector(np.cross(axes_new[0], axes_new[1]))])
                print("axes1", axes)
        floor_normal = axes[2]
        floor_normal = -np.sign(floor_normal[2])*floor_normal

    print("fov", fov, floor_normal)
    new_cam, _ = camera_fov_res_to_intrinsics(fov, np.array([img.shape[1], img.shape[0]]))

    plane, depth = calcPlaneXYZ([floor_normal*floor_offset], width=img.shape[1], height=img.shape[0], camera=new_cam, max_depth=10)
    plane = plane[0]

    fm = cv2.resize(floor_mask, (img.shape[1], img.shape[0]))

    plane_center = np.mean(plane[fm>.5], axis=0)
    floor_offset = np.dot(plane_center, floor_normal)

    basis_forward = geometry.unit_vector(np.float32([0, floor_normal[2], -floor_normal[1]]))
    basis_right = geometry.unit_vector(np.cross(basis_forward, floor_normal))

    floor_rotation = 0
    if axes[1] is not None:
        floor_rotation = -geometry.angle_between(axes[1], np.sign(basis_forward[1] - axes[1,1]) * np.sign(
        basis_forward[0] - axes[1,0])*basis_forward)

    img_dir = img.copy()
    # M = np.eye(3)
    # from cambrian import utils
    #
    # M = utils.axisAngleToRotationMatrix(floor_normal,floor_rotation)
    #
    # basis_right = np.dot(M, basis_right)
    # basis_forward = np.dot(M, basis_forward)
    #
    # uv_center = np.int32(xyz_to_uv(plane_center, width=img.shape[1], height=img.shape[0], camera=new_cam))
    # pre_pos = np.int32(xyz_to_uv(plane_center, width=img.shape[1], height=img.shape[0], camera=new_cam))
    # color = np.int32(np.random.randint([127, 127, 127], [254, 254, 235]))
    # color = (int(color[0]), int(color[1]), int(color[2]))
    # mask = np.zeros_like(img_dir)
    # a = 50
    # scale = 4.
    #
    #
    # # plane_center = plane_center +.33 * basis_right-.68*basis_forward
    # print(plane_center)
    # for l in range(-a, a):
    #     for k in range(-a, a):
    #
    #         pos = plane_center + l / scale * basis_right + k / scale * basis_forward
    #         pos2 = plane_center + (l + 1) / scale * basis_right + (k - 1) / scale * basis_forward
    #
    #         uv_pos = np.int32(xyz_to_uv(pos, width=img.shape[1], height=img.shape[0], camera=new_cam))
    #         uv_pos2 = np.int32(xyz_to_uv(pos2,width=img.shape[1], height=img.shape[0], camera=new_cam))
    #
    #         if np.isnan(uv_pos[0]) or np.isnan(uv_pos[1]) or np.isnan(uv_pos2[1]) or np.isnan(uv_pos2[0]): continue
    #         if k < a - 1 and k >= -a + 1 and l < a - 1 and l >= -a + 1:
    #             cv2.circle(mask, (int(plane_center[0]), int(plane_center[1])), 3, color, thickness=-1)
    #             cv2.circle(mask, (uv_pos[0], uv_pos[1]), 3, color, thickness=-1)
    #             cv2.line(mask, (pre_pos[0], pre_pos[1]), (uv_pos[0], uv_pos[1]), color=(255, 0, 255), thickness=2)
    #             cv2.line(mask, (pre_pos[0], pre_pos[1]), (uv_pos2[0], uv_pos2[1]), color=(0, 255, 255),
    #                      thickness=2)
    #         pre_pos = uv_pos
    # img_dir[np.logical_and(mask[:, :, 2] > 0, fm>.5)] = mask[np.logical_and(mask[:, :, 2] > 0, fm>.5)]
    #
    # color_index = 0
    # for i in axes_indices:
    #     uv_center = pp
    #     vp =vps[i]/vps[i, 2]
    #     color = [0,0,0]
    #     color[color_index] = 255
    #     color_index += 1
    #     cv2.polylines(img_dir, pts=np.array([[[uv_center[0], uv_center[1]], [vp[0], vp[1]]]], np.int32),
    #                   isClosed=False, color=color, thickness=2)

    # save_dir = '/Users/derrickhart/cb-base/client-visualizers/cambrianar-sites/divinefloor/scenes/bedroom/2-bedroom/'
    # cv2.imwrite(save_dir + "image.jpg", img_dir)

    return vps, inliers, floor_normal, floor_offset, floor_rotation, fov, img_dir


class PipelineRefinePlaneMasks(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "hed", "mask", "floor_rotation"]

    @property
    def output_keys(self) -> list:
        return ["planes", "mask", "lighting", "floor_rotation"]

    def run(self, data):

        if IM_LOGGING_ENABLED:
            with open('logging/data.pickle', 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        logging_index = 0
        logging_dir = 'logging/'

        img = data["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[1] > 1024:
            img = cv2.resize(img, (1024, int(img.shape[0] / img.shape[1] * 1024)))

        logging_index = _log_image(logging_dir, "image.png", img, logging_index=logging_index)

        output = np.float32(data["semantic_probs"])

        hed = data["hed"]

        hed_lr = hed
        logging_index = _log_image(logging_dir, 'hed.png', hed_lr, logging_index=logging_index)

        h, w = output[0].shape

        shape = (w, h)

        camera = camera_fov_res_to_intrinsics(data["fov"], np.array(shape))

        hed_lr = cv2.resize(hed, shape)

        img_lr = cv2.resize(img, shape)

        sx = w / img.shape[1]
        sy = h / img.shape[0]

        rug = output[28]
        output[3] += rug
        output[28] = 0
        output[3] += output[13]
        output[13] = 0
        output[3] += output[9]
        output[9] = 0

        wall_mask = output[0].copy()

        logging_index = _log_image(logging_dir, "wall_mask.png", 255. * (wall_mask), logging_index=logging_index)

        floor_mask = output[3].copy()
        logging_index = _log_image(logging_dir, "floor_mask.png", 255. * floor_mask, logging_index=logging_index)

        ceiling_mask = output[5].copy()
        logging_index = _log_image(logging_dir, "ceiling_mask.png", 255. * ceiling_mask, logging_index=logging_index)

        wall_like_mask = np.zeros_like(wall_mask)

        for i in wall_like:
            if i == 0: continue
            wall_like_mask += output[i]

        logging_index = _log_image(logging_dir, "wall_like_mask.png", 255. * wall_like_mask,
                                   logging_index=logging_index)

        other_mask = 1.0 - floor_mask - wall_mask - wall_like_mask - ceiling_mask

        logging_index = _log_image(logging_dir, "other_mask.png", 255. * other_mask, logging_index=logging_index)

        ade_seg_c = np.dstack(
            (.95 * np.ones_like(other_mask), other_mask, floor_mask, wall_mask, ceiling_mask, wall_like_mask))

        ade_seg = np.argmax(ade_seg_c, -1)
        ade_skel = skeletonize(other_mask > .5)

        logging_index = _log_segmentation_image(logging_dir, "ade_seg.png", np.int32(ade_seg), img_lr,
                                                logging_index=logging_index)

        line_data, lines = find_lines(img, cv2.resize(hed, (img.shape[1], img.shape[0])), data["normals"])

        all_lines = np.int32(np.zeros((img_lr.shape[0], img_lr.shape[1])))

        l_image_rgb = img_lr.copy()
        l = 1

        for line in lines:
            for x1, y1, x2, y2 in line:
                x1 = int(sx * x1)
                x2 = int(sx * x2)
                y1 = int(sy * y1)
                y2 = int(sy * y2)
                cv2.line(all_lines, (x1, y1), (x2, y2), l, thickness=2, lineType=cv2.LINE_8)
                l += 1

        merged_lines = np.int32(np.zeros((img_lr.shape[0], img_lr.shape[1])))
        Line.draw_all(line_data, merged_lines, color=255, thickness=2, sx=sx, sy=sy, lineType=cv2.LINE_4)

        l_image_rgb[merged_lines > 0] = 255
        logging_index = _log_segmentation_image(logging_dir, "l_image.png", all_lines, img_lr,
                                                logging_index=logging_index)

        logging_index = _log_image(logging_dir, "l_image_rgb.png", l_image_rgb, logging_index=logging_index)

        planes_data = data["planes"]
        plane_parameters = np.array(data["planes"]["detection"][:, 6:9], dtype=np.float32)
        plane_offsets = np.linalg.norm(plane_parameters, axis=-1, keepdims=True)
        plane_normals = plane_parameters / np.maximum(plane_offsets, 1e-4)
        plane_clusters = np.array(data["planes"]["detection"][:, 4], dtype=np.int32)
        # roi = np.array(data["planes"]["detection"][:, 0:4], dtype=np.int32)
        cluster_prob = data["planes"]["detection"][:, 5]

        plane_masks = planes_data["masks"]
        number_planes = len(plane_masks)
        plane_rotations = np.zeros(number_planes, dtype=np.float32)
        floor_rotation = 0.0

        plane_XYZ = planes_data["plane_XYZ"][:, :, 80:-80, :].transpose(0, 2, 3, 1)

        plane_masks = resize_array(plane_masks, shape)
        plane_XYZ = resize_array(plane_XYZ, shape)

        # depth = planes_data["depth_np"][:, 80:-80, :].transpose(1, 2, 0)
        XYZ = planes_data["XYZ"][:, 80:-80, :].transpose(1, 2, 0)
        # depth = cv2.resize(depth, shape)
        XYZ = cv2.resize(XYZ, shape)
        logging_index = _log_image(logging_dir, "XYZ.png", 255. * XYZ / np.amax(XYZ), logging_index=logging_index)

        normals = cv2.resize(data["normals"], shape)
        logging_index = _log_image(logging_dir, "normals.png", normals, logging_index=logging_index)

        normals = (normals - 127.5) / 127.5

        floor_indices, ceiling_indices, horiz_indices, floor_angs = find_floor_indices(floor_mask, ceiling_mask,
                                                                                       plane_masks, plane_normals)

        # print("floor indices are: ", floor_indices, horiz_indices)
        # print("ceiling indices are: ", ceiling_indices)

        floor_normal = [0., 0, -1]
        floor_index = -1

        if len(floor_indices) > 0:
            floor_index = floor_indices[0]

            floor_normal = plane_normals[floor_index]
            floor_offset = plane_offsets[floor_index]
        # print("floor_normal", floor_normal)

        ceiling_index = -1

        if len(ceiling_indices) > 0:
            ceiling_index = ceiling_indices[0]

        wall_indices, vert_indices, vert_angs = find_wall_indices(wall_mask, plane_masks, plane_normals[floor_index],
                                                                  plane_normals)

        # print("wall indices are: ", wall_indices, vert_indices, vert_angs)

        basis_indices = np.int32(np.concatenate([floor_indices, ceiling_indices, wall_indices]))

        ceiling_normal = plane_normals[ceiling_index]
        # print("ceiling normal, floor normal", ceiling_normal, floor_normal, np.dot(ceiling_normal,floor_normal))

        normals_combined, normals_nn_normals = combined_normals(-normals, plane_normals, plane_masks, basis_indices,
                                                                cluster_prob)
        normals_c = normals_combined

        lengths = np.maximum(np.sqrt(np.sum(normals_c * normals_c, -1)), 1e-6)
        normals_c /= np.dstack((lengths, lengths, lengths))
        logging_index = _log_image(logging_dir, "normals_c_org.png", 127.5 * (normals_c + 1),
                                   logging_index=logging_index)

        cluster_masks = [.03 * np.ones_like(plane_masks[0])]
        cluster_mask_indices = [0]
        for i in range(1, 8):
            clust = np.nonzero(plane_clusters == i)[0]
            clust = np.intersect1d(clust, vert_indices)

            if len(clust) > 0:
                cluster_masks.append(np.sum(plane_masks[clust], 0))
                cluster_mask_indices.append((i - 1) % 3 + 1)

        plane_cluster_seg = np.argmax(cluster_masks, 0)
        cluster_mask_indices = np.int32(cluster_mask_indices)
        plane_cluster_seg_rs = np.int32(
            cv2.resize(np.uint8(plane_cluster_seg), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST))

        logging_index = _log_segmentation_image(logging_dir, "plane_cluster_seg.png", plane_cluster_seg, img_lr,
                                                logging_index=logging_index)

        logging_index = _log_image(logging_dir, "XYZ.png", 255. * XYZ / np.amax(XYZ), logging_index=logging_index)
        logging_index = _log_ply(logging_dir, "3D.ply", img_lr, plane_masks, np.float32(plane_XYZ), mult=1,
                                 logging_index=logging_index)

        ######################################## Initial refinement work
        w_mask = merged_lines == 0

        other_markers = np.int32(
            refine_surface(other_mask, img_lr, big_thresh=.001, small_thresh=.95, watershed_dist=.03,
                           gradient=False))

        wall_markers = np.int32(
            refine_surface(wall_mask, hed_lr, big_thresh=.05, small_thresh=.95, watershed_dist=.05, gradient=True,
                           watershed_mask=w_mask))

        floor_markers = np.int32(
            refine_surface(floor_mask, img_lr, big_thresh=.001, small_thresh=.95, watershed_dist=.05, gradient=False))

        wall_like_markers = np.int32(
            refine_surface(wall_like_mask, img_lr, big_thresh=.001, small_thresh=.95, watershed_dist=.05,
                           gradient=False))

        ceiling_markers = np.int32(
            refine_surface(ceiling_mask, hed_lr, big_thresh=.05, small_thresh=.95, watershed_dist=.05, gradient=True,
                           watershed_mask=w_mask))

        ceiling_prob = get_segmentation_image(ceiling_markers + 1, ceiling_mask, avg=True)
        wall_like_prob = get_segmentation_image(wall_like_markers + 1, wall_like_mask, avg=True)
        wall_like_prob[wall_like_prob < .25] = 0
        wall_prob = get_segmentation_image(wall_markers + 1, wall_mask, avg=True)
        floor_mask[ade_skel > 0] = 0
        floor_prob = get_segmentation_image(floor_markers + 1, floor_mask, avg=True)
        floor_prob[floor_prob < .25] = 0
        floor_markers[floor_prob < .25] = 100
        other_markers = join_segmentations(floor_markers, other_markers)

        other_mask[ade_skel > 0] = 1
        other_prob = get_segmentation_image(other_markers + 1, other_mask, avg=True)

        logging_index = _log_image(logging_dir, "floor_markers.png", 255. * floor_prob, logging_index=logging_index)
        logging_index = _log_image(logging_dir, "other_markers.png", 255. * other_prob, logging_index=logging_index)
        logging_index = _log_image(logging_dir, "ceiling_markers.png", 255. * ceiling_prob, logging_index=logging_index)
        logging_index = _log_image(logging_dir, "wall_like_markers.png", 255. * wall_like_prob,
                                   logging_index=logging_index)
        logging_index = _log_image(logging_dir, "wall_markers.png", 255. * wall_prob, logging_index=logging_index)

        segmentation_initial = np.int32(np.argmax(np.dstack(
            (.05 * np.ones_like(other_mask), other_prob, floor_prob, wall_prob, ceiling_prob, wall_like_prob)), -1))

        segmentation_initial[np.logical_and(segmentation_initial == 3, wall_prob < .5)] = 6
        m = np.logical_and(segmentation_initial == 2, other_prob > .5)
        segmentation_initial[merged_lines > 0] = 0
        segmentation_initial[m] = 1

        for i in range(1, 7):
            pruned = remove_small_objects(segmentation_initial == i, min_size=32)
            segmentation_initial[np.logical_and(segmentation_initial == i, pruned == 0)] = 0

        line_mask = Line.draw_all(line_data,
                                  np.zeros((segmentation_initial.shape[0], segmentation_initial.shape[1])),
                                  color=255,
                                  thickness=2, sx=sx, sy=sy, lineType=cv2.LINE_4)

        segmentation_initial = watershed(hed_lr, segmentation_initial,
                                         mask=line_mask == 0)
        segmentation_initial = cv2.watershed(img_lr, segmentation_initial)

        segmentation_initial[line_mask > 0] = 0
        distances = cv2.distanceTransform(np.uint8(line_mask), cv2.DIST_L1, 3)

        distances = np.uint8(distances)
        segmentation_initial = cv2.watershed(cv2.cvtColor(np.uint8(distances), cv2.COLOR_GRAY2BGR),
                                             segmentation_initial)

        logging_index = _log_segmentation_image(logging_dir, "segmentation_initial.png", segmentation_initial, img_lr,
                                                logging_index=logging_index)

        #############################################################

        sure_walls = segmentation_initial == 3
        sure_floor = segmentation_initial == 2
        sure_wall_like = np.logical_or(sure_walls, segmentation_initial == 5)

        edgelets = compute_edgelets(lines)

        vps, inliers, floor_normal, floor_offset, floor_rotation, fov, img_dir = compute_normal_from_vps(edgelets, img,
                                                                                                         data["fov"],
                                                                                                         floor_normal,
                                                                                                         floor_offset,
                                                                                                         floor_mask)

        data["fov"] = fov

        if floor_index > -1:
            plane_parameters[floor_indices[0]] = floor_normal * floor_offset

        logging_index = _log_image(logging_dir, "img_dir.png", img_dir,
                                   logging_index=logging_index)

        vp0 = vps[0] / vps[0][2]

        vertical_line_inliers = inliers[0]
        locations, directions, strengths = edgelets

        edgelets = (
            locations[vertical_line_inliers], directions[vertical_line_inliers], strengths[vertical_line_inliers])

        locations, directions, strengths = edgelets

        vp_directions = locations - vp0[:2]

        # arrange lines from left to right relative to vertical vp
        angles = np.arctan2(vp_directions[:, 1], vp_directions[:, 0])

        s = np.argsort(np.sign(vp0[1]) * angles)

        locations = locations[s]

        vp0[:2] *= [sx, sy]
        locations[:, 0] *= sx
        locations[:, 1] *= sy
        camera, _ = camera_fov_res_to_intrinsics(fov, np.array(shape))

        # Create fan from vertical vp

        labels_fan = np.int32(np.zeros((img_lr.shape[0], img_lr.shape[1])))

        fan_normals = []

        k = 1
        for i in range(-1, len(locations)):
            if i == -1:
                closest = np.int32([[0, img_lr.shape[1]], [0, 0]])
                dir = closest - vp0[:2]
                closest_index = np.argmin(np.abs(dir[:, 1]))
                pt1 = np.int32(2 * closest[closest_index] - vp0[:2])
                pt2 = np.int32(2 * locations[i + 1] - vp0[:2])
            elif i == len(locations) - 1:
                closest = np.int32([[img_lr.shape[1], img_lr.shape[0]], [img_lr.shape[1], 0]])
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
            wedge = np.logical_and(wall_arc, wall_mask > .9)
            a = np.sum(wedge)

            if a > 0:
                labels_fan[wall_arc > 0] = k

                cur_normal = np.mean(normals_c[wedge], 0)
                cur_normal /= max(np.linalg.norm(cur_normal), .00001)
                fan_normals.append(cur_normal)
                k += 1
            # else:
            #     if len(fan_normals)>0:
            #         fan_normals.append(fan_normals[-1])

        # Merge by normal angle diff

        labels_fan, fan_normals_reduced, normals_wall = merge_by_angle_sweep(labels_fan, normals_c, fan_normals,
                                                                             wall_mask > .9, angle_threshold=.85)

        logging_index = _log_segmentation_image(logging_dir, "fan1.png", labels_fan, img_lr,
                                                logging_index=logging_index)

        unique_labels = np.unique(labels_fan[labels_fan > 0])

        fan_normals_reduced = np.float32([np.mean(normals_c[labels_fan == j], 0) for j in unique_labels])
        lengths = np.sqrt(np.sum(fan_normals_reduced * fan_normals_reduced, -1))
        fan_normals_reduced /= np.dstack((lengths, lengths, lengths))[0]

        labels_fan, fan_normals_reduced, normals_wall = merge_by_angle_sweep(labels_fan, normals_wall,
                                                                             fan_normals_reduced, wall_mask > 0,
                                                                             angle_threshold=.8)

        vl_image = np.zeros_like(all_lines)
        vl_image[sure_walls == 0] = 0
        logging_index = _log_segmentation_image(logging_dir, "fan2.png", labels_fan, img_lr,
                                                logging_index=logging_index)
        logging_index = _log_image(logging_dir, "normals_wall_org.png", 127.5 * (normals_wall + 1),
                                   logging_index=logging_index)
        logging_index = _log_segmentation_image(logging_dir, "vl_image.png", vl_image, img_lr,
                                                logging_index=logging_index)

        # logging_index = _log_ply(logging_dir, "3D2.ply", img_lr, plane_masks, np.float32(plane_XYZ), mult=1, logging_index = logging_index)

        plane_classes = get_planes_class(plane_masks, ade_seg)
        # print("plane_classes", plane_classes)

        full_planes = plane_masks.copy()
        full_planes[full_planes < .01] = 0

        plane_masks[plane_masks < .5] = 0
        wall_like_planes = [np.zeros_like(hed_lr / 255.)]

        wall_like_indices = []
        floor_indices = []
        ceiling_indices = []

        for k in range(number_planes):
            # print("cluster", cluster_prob[k])
            a = np.int32(plane_classes[k])

            name = "plane_" + str(k)
            name += "_" + str(a)

            if a == 3 or a == 5:
                if k not in wall_like_indices:
                    wall_like_indices.append(k)
                    if cluster_prob[k] > .5:
                        wall_like_planes.append(plane_masks[k])
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

        logging_index = _log_segmentation_image(logging_dir, "wall_planes_seg.png", wall_planes_seg, img_lr,
                                                logging_index=logging_index)

        labels_wall_1 = labels_fan
        labels_wall_1[labels_wall_1 > 0] += np.amax(wall_planes_seg) + 1
        logging_index = _log_segmentation_image(logging_dir, "labels_wall_graph.png", labels_fan, img_lr, avg=False,
                                                logging_index=logging_index)

        for i in np.unique(labels_wall_1):
            if i == 0: continue

            mask = i == labels_wall_1
            m = mode(wall_planes_seg[np.logical_and(wall_planes_seg > 0, mask)])

            if len(m[0]) > 1:
                labels_wall_1[mask] = m[0]

        logging_index = _log_segmentation_image(logging_dir, "labels_wall_merge1.png", labels_wall_1, img_lr, avg=False,
                                                logging_index=logging_index)

        label_indices = np.unique(labels_wall_1)

        wall_planes_number = len(wall_like_indices)

        means = np.zeros((len(label_indices), wall_planes_number + 1), dtype=np.float32)

        for i in range(1, wall_planes_number + 1):
            means[:, i] = ndimage.mean(full_planes[wall_like_indices[i - 1]], labels=labels_wall_1, index=label_indices)

        means[means < .05] = 0
        arg = np.argmax(means, axis=-1)

        labels_arg = np.zeros_like(np.int32(labels_wall_1))

        vert_not_wall = np.int32(np.setdiff1d(vert_indices, wall_like_indices))
        all_vertical = np.int32(np.union1d(vert_indices, wall_like_indices))

        for i in range(1, len(label_indices)):
            label_mask = labels_wall_1 == label_indices[i]
            if arg[i] > 0:
                labels_arg[label_mask] = wall_like_indices[arg[i] - 1] + 1

            else:
                print('we look for a vertical plane instead of wall')
                vert_means = np.mean(full_planes[vert_not_wall][:, label_mask])
                vert_arg = np.argmax(vert_means, -1)

                if vert_means > .01:
                    labels_arg[label_mask] = vert_not_wall[vert_arg] + 1
                    plane_center = np.mean(XYZ[label_mask], axis=0)
                    offset = np.dot(plane_center, plane_normals[vert_not_wall[vert_arg]])
                    # print(all_vertical[wall_index], plane_offsets[all_vertical[wall_index]])
                    plane_parameters[vert_not_wall[vert_arg]] = plane_normals[vert_not_wall[vert_arg]] * offset
                else:
                    label_normal = np.mean(normals_wall[label_mask], 0)

                    label_normal /= max(np.linalg.norm(label_normal), .00001)
                    if len(all_vertical) > 0:
                        wall_index = np.argmax(np.dot(plane_normals[all_vertical], label_normal))

                        plane_center = np.mean(XYZ[label_mask], axis=0)
                        offset = np.dot(plane_center, label_normal)

                        plane_parameters[all_vertical[wall_index]] = plane_normals[all_vertical[wall_index]] * offset

                        labels_arg[label_mask] = all_vertical[wall_index] + 1
                        print("if no good match just take the closest by angle", all_vertical,
                              plane_parameters[all_vertical[wall_index]], all_vertical[wall_index])

        plane_XYZ, plane_depth = calcPlaneXYZ(plane_parameters, width=w, height=h, camera=camera, max_depth=10)
        logging_index = _log_segmentation_image(logging_dir, 'labels_arg.png', labels_arg, img_lr,
                                                logging_index=logging_index)

        # needs to be cleaned up
        # labels_wall_1 = labels_arg.copy()

        labels_wall_2 = labels_arg.copy()
        labels_wall_2[labels_wall_2 > 0] += np.amax(wall_planes_seg)

        for i in np.unique(labels_arg):
            if i == 0: continue

            label_mask = labels_arg == i

            intersection = np.sum(label_mask[wall_planes_seg > 0])
            # print("intersecxtion", intersection)
            if intersection > 1:
                # print(np.unique(wall_planes_seg[label_mask>0]))
                w = watershed(hed_lr, wall_planes_seg, mask=label_mask)
                # print(np.unique(w))
                l = np.logical_and(w > 0, label_mask > 0)
                u = np.unique(w[l])

                color_means = [np.mean(img_lr[w == s], 0) for s in u]
                color_n = len(color_means)

                for cm1 in range(color_n):
                    for cm2 in range(cm1 + 1, color_n):
                        color_measure = np.linalg.norm(color_means[cm1] - color_means[cm2])
                        # print("color diff", color_measure)
                        if color_measure < 30:
                            w[w == u[cm2]] = u[cm1]

                labels_wall_2[l] = w[l]

        logging_index = _log_segmentation_image(logging_dir, "labels_wall_merge2.png", labels_wall_2, img_lr, avg=False,
                                                logging_index=logging_index)

        final_masks = []

        final_plane_parameters = []
        final_rotations = []
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

                plane_parameter = np.zeros(11)
                plane_parameter[:9] = data["planes"]["detection"][l][:9]
                plane_parameter[6:9] = plane_parameters[l]
                plane_parameter[9] = 2
                plane_parameter[10] = plane_rotations[l]

                # _log_image(logging_dir, str(l) + "plane_masks.png", 255. * plane_masks[l])

                final_plane_parameters.append(plane_parameter)
                final_plane_XYZ.append(plane_XYZ[l])
                final_rotations.append(plane_rotations[l])

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

            plane_parameter = np.zeros((11))
            plane_parameter[:9] = data["planes"]["detection"][floor_indices[0]][:9]
            plane_parameter[6:9] = plane_parameters[floor_indices[0]]
            plane_parameter[9] = 1
            plane_parameter[10] = plane_rotations[floor_indices[0]]

            final_plane_parameters.append(plane_parameter)
            final_plane_XYZ.append(plane_XYZ[floor_indices[0]])
            final_rotations.append(plane_rotations[floor_indices[0]])

        # add ceiling
        if len(ceiling_indices) > 0:
            ceiling_mask = 255 * np.uint8(segmentation_initial == 4)
            final_masks.append(ceiling_mask)
            plane_parameter = np.zeros((11))
            plane_parameter[:9] = data["planes"]["detection"][ceiling_indices[0]][:9]
            plane_parameter[6:9] = plane_parameters[ceiling_indices[0]]
            plane_parameter[9] = 3
            plane_parameter[10] = plane_rotations[ceiling_indices[0]]
            final_plane_parameters.append(plane_parameter)
            final_plane_XYZ.append(plane_XYZ[ceiling_indices[0]])
            final_rotations.append(plane_rotations[ceiling_indices[0]])

        final_plane_parameters = np.float32(final_plane_parameters)
        final_rotations = np.float32(final_rotations)

        final_plane_number = len(final_masks)

        # print("final_plane_number", final_plane_number)
        final_masks = np.uint8(final_masks)

        final_labels = np.argmax(final_masks, 0)

        final_labels[final_labels > 0] += 1
        final_labels[final_masks[0] > 0] = 1
        final_labels += 1
        final_labels = np.uint8(final_labels)

        lighting_rgb = np.uint8(data["lighting"])

        sigma_r = 1.0
        sigma_s = 10

        lighting_smooth = cv2.edgePreservingFilter(lighting_rgb, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)
        logging_index = _log_image(logging_dir, 'lighting_smooth.png', lighting_smooth, logging_index=logging_index)

        data["lighting"] = lighting_smooth

        logging_index = _log_image(logging_dir, 'lighting.png', lighting_smooth, logging_index=logging_index)

        length_threshold = 32
        canny_aperture_size = 7

        fld = cv2.ximgproc.createFastLineDetector(_length_threshold=length_threshold,
                                                  _canny_aperture_size=canny_aperture_size)
        final_labels[vl_image > 0] = 0
        lines = fld.detect(final_labels)

        final_line_data = []
        for line in line_data:
            # t = final_labels_hr[int(line.midpoint[1]),int(line.midpoint[0])]
            # if t==0:
            final_line_data.append(line)

        if lines is not None:
            for line in lines:
                new_line = Line(line[0][0] / sx, line[0][1] / sy, line[0][2] / sx, line[0][3] / sy)
                if new_line.get_confidence() > 0.1: final_line_data.append(new_line)

        final_line_data = Line.merge(final_line_data, 3, search_length=1.01, angle_threshold=math.radians(1.0))

        mask_res = 2048
        mask_shape = (mask_res, int(mask_res * img.shape[0] / img.shape[1]))

        if img.shape[0] > img.shape[1]:
            mask_shape = (int(mask_res * img.shape[1] / img.shape[0]), mask_res)

        final_masks_hr = resize_array(np.uint8(final_masks), mask_shape)
        final_labels_hr = np.int32(np.argmax(final_masks_hr, 0))

        final_labels_hr[final_labels_hr > 0] += 1
        final_labels_hr[final_masks_hr[0] > 0] = 1
        final_labels_hr += 1

        final_merged_lines = Line.draw_all(final_line_data, np.zeros_like(final_labels_hr), color=1, thickness=6,
                                           sx=final_labels_hr.shape[1] / img.shape[1],
                                           sy=final_labels_hr.shape[0] / img.shape[0],
                                           lineType=cv2.LINE_AA)

        edges = cv2.Canny(cv2.cvtColor(
            cv2.GaussianBlur(cv2.resize(img, (final_merged_lines.shape[1], final_merged_lines.shape[0])), (3, 3), 2),
            cv2.COLOR_BGR2GRAY), 100, 200)

        final_merged_lines[edges > 0] = 1
        final_labels_hr[final_merged_lines > 0] = 0

        final_labels_hr = np.uint8(final_labels_hr)

        distances = cv2.distanceTransform(np.uint8(final_merged_lines), cv2.DIST_L1, 3)
        distances = np.uint8(distances)

        final_labels_hr = ip.refine_mask_watershed(None, cv2.resize(img, (
            final_merged_lines.shape[1], final_merged_lines.shape[0])),
                                                   final_labels_hr, None, distance=0.01) - 1

        for i in np.unique(final_labels_hr):
            mask = final_labels_hr == i
            # inter = np.logical_and(mask, final_merged_lines)
            # final_labels_hr[mask > 0] = 0
            # mask[final_merged_lines > 0] = 0
            pruned = remove_small_objects(mask, 100)  # pruned[inter > 0] = 1
            # filled = np.uint8(remove_small_holes(mask,10000))
            final_labels_hr[mask > 0] = 0
            final_labels_hr[pruned > 0] = i

        logging_index = _log_segmentation_image(logging_dir, "pre_final_labels.png", final_labels_hr - 1, img,
                                                logging_index=logging_index)

        final_labels_hr = cv2.watershed(cv2.cvtColor(distances, cv2.COLOR_GRAY2BGR), np.int32(final_labels_hr))
        final_labels_hr[final_labels_hr < 0] = 0
        final_labels = np.int32(final_labels_hr)

        data["planes"]["masks"] = np.zeros((final_plane_number, final_labels.shape[0], final_labels.shape[1]),
                                           dtype=np.uint8)

        data["planes"]["detection"] = np.zeros((final_plane_number, 11), dtype=data["planes"]["detection"].dtype)
        data["planes"]["detection"] = final_plane_parameters
        # data["planes"]["rotation"] = final_rotations

        mask_contours = []

        for d in range(final_plane_number):
            contours, hierarchy = cv2.findContours(np.uint8(final_labels == d + 2), cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            data["planes"]["masks"][d] = np.zeros_like(np.uint8(final_labels == d + 2))

            plane_contours = []

            if len(contours) > 0:

                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])

                    if area > 4 * 16 * 16:
                        if hierarchy[0, i, 3] == -1:  # this is the outer contour which we need to draw
                            cv2.drawContours(data["planes"]["masks"][d], [contours[i]], -1, 255, -1, cv2.LINE_AA)
                            # cv2.drawContours(data["planes"]["masks"][d], [contours[i]], -1, 255, 4,cv2.LINE_AA)
                            # cv2.drawContours(final_labels, [contours[i]], -1, d + 2, -1, cv2.LINE_AA)
                            # cv2.drawContours(final_labels, [contours[i]], -1, d + 2, 4, cv2.LINE_AA)
                            plane_contours.append(np.array([[[0, 0]]]).tolist())

                        else:
                            cv2.drawContours(data["planes"]["masks"][d], contours, i, 0, -1)

                # rect = cv2.boundingRect(data["planes"]["masks"][d])

                data["planes"]["detection"][d, 0:4] = [
                    0, 0, 0, 0
                ]

            mask_contours.append(plane_contours)

            data["planes"]["contours"] = mask_contours

        # get rid of this later
        data["mask"] = np.zeros_like(data["planes"]["masks"][0])
        if len(data["planes"]["masks"]) > 2:
            data["mask"] = data["planes"]["masks"][-2]

        if abs(floor_rotation) > 0:
            data["floor_rotation"] = floor_rotation
        logging_index = _log_segmentation_image(logging_dir, "final_labels.png", np.int32(final_labels) - 1, img,
                                                logging_index=logging_index)
        # print("floor rotation check", data["floor_rotation"])