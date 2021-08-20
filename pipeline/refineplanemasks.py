from pipeline.core import PipelineStep
import cv2
import numpy as np
from scipy import ndimage
from time import time
import os

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

from pipeline.utils import resize_array
from pipeline.planegeometry import PlaneGeometry, SemanticKey, Dimension
from skimage.morphology import remove_small_objects, remove_small_holes
from pipeline.semanticlabels import ADE20K
from pipeline.logging import get_segmentation_image, log_image, log_segmentation_image, log_ply, im_logging_enabled, LogLevel
from pipeline.fovestimator import compute_edgelets, compute_normal_from_vps, calcPlaneXYZ, camera_fov_res_to_intrinsics

from enum import Enum

furniture_labels = [ADE20K.table, ADE20K.armchair, ADE20K.sofa, ADE20K.coffee_table, ADE20K.ottoman, ADE20K.chest, ADE20K.wardrobe, ADE20K.chair, ADE20K.bed, ADE20K.bench, ADE20K.swivel_chair, ADE20K.pole, ADE20K.stool]
wall_like = [ADE20K.windowpane, ADE20K.door, ADE20K.curtain, ADE20K.painting, ADE20K.shelf, ADE20K.column, ADE20K.screen_door, ADE20K.blind, ADE20K.projection_screen]

f=1.0
METADATA = np.array([571.87, 571.87, 320, 240, 640, 480, 0, 0, 0, 0])

IMAGE_MAX_DIM = 640
IMAGE_MIN_DIM = 480


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


#######################################

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

#labels for ade20k, subtract = 1 for output number. 

class PipelineRefinePlaneMasks(PipelineStep):
    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "hed", "mask", "floor_rotation"]

    @property
    def output_keys(self) -> list:
        return ["planes", "mask", "lighting", "floor_rotation"]

    def _combine_floor_masks(self, output):
        
        output[ADE20K.floor.index] += output[ADE20K.rug.index]
        output[ADE20K.rug.index] = 0

        output[ADE20K.floor.index] += output[ADE20K.earth.index]
        output[ADE20K.earth.index] = 0

        output[ADE20K.floor.index] += output[ADE20K.grass.index]
        output[ADE20K.grass.index] = 0


    def _isolate_masks(self, data, output):

        isolated = {}

        isolated[SemanticKey.Wall] = output[ADE20K.wall.index].copy()
        isolated[SemanticKey.Floor] = output[ADE20K.floor.index].copy()
        isolated[SemanticKey.Ceiling] = output[ADE20K.ceiling.index].copy()
        isolated[SemanticKey.WallLike] = np.zeros_like(isolated[SemanticKey.Wall])

        for label in wall_like:
            isolated[SemanticKey.WallLike] += output[label.index]

        isolated[SemanticKey.Other] = 1.0 - isolated[SemanticKey.Floor] - isolated[SemanticKey.Wall] - isolated[SemanticKey.WallLike] - isolated[SemanticKey.Ceiling]

        if im_logging_enabled(data, LogLevel.Segmentation):
            for key in isolated.keys():
                log_image(data, key.value, 255. * isolated[key])

        return isolated

    def _get_lines_image(self, data, img, lines, sx, sy):
        all_lines = np.int32(np.zeros((img.shape[0], img.shape[1])))
        l = 1

        for line in lines:
            for x1, y1, x2, y2 in line:
                x1 = int(sx * x1)
                x2 = int(sx * x2)
                y1 = int(sy * y1)
                y2 = int(sy * y2)
                cv2.line(all_lines, (x1, y1), (x2, y2), l, thickness=2, lineType=cv2.LINE_8)
                l += 1

        return all_lines

    def _refine_surfaces(self, data, isolated, line_data, img_lr, hed_lr, sx, sy):
        other_markers = np.int32(
            refine_surface(isolated[SemanticKey.Other], img_lr, big_thresh=.001, small_thresh=.95, watershed_dist=.03, gradient=False))

        wall_markers = np.int32(
            refine_surface(isolated[SemanticKey.Wall], hed_lr, big_thresh=.05, small_thresh=.95, watershed_dist=.05, gradient=True))

        floor_markers = np.int32(
            refine_surface(isolated[SemanticKey.Floor], img_lr, big_thresh=.001, small_thresh=.95, watershed_dist=.05, gradient=False))

        wall_like_markers = np.int32(
            refine_surface(isolated[SemanticKey.WallLike], img_lr, big_thresh=.001, small_thresh=.95, watershed_dist=.05, gradient=False))

        ceiling_markers = np.int32(
            refine_surface(isolated[SemanticKey.Ceiling], hed_lr, big_thresh=.05, small_thresh=.95, watershed_dist=.05, gradient=True))

        ceiling_prob = get_segmentation_image(ceiling_markers + 1, isolated[SemanticKey.Ceiling], avg=True)
        wall_like_prob = get_segmentation_image(wall_like_markers + 1, isolated[SemanticKey.WallLike], avg=True)
        wall_like_prob[wall_like_prob < .25] = 0
        wall_prob = get_segmentation_image(wall_markers + 1, isolated[SemanticKey.Wall], avg=True)

        ade_skel = skeletonize(isolated[SemanticKey.Other] > .5)
        isolated[SemanticKey.Floor][ade_skel > 0] = 0
        floor_prob = get_segmentation_image(floor_markers + 1, isolated[SemanticKey.Floor], avg=True)
        floor_prob[floor_prob < .25] = 0
        floor_markers[floor_prob < .25] = 100
        other_markers = join_segmentations(floor_markers, other_markers)

        isolated[SemanticKey.Other][ade_skel > 0] = 1
        other_prob = get_segmentation_image(other_markers + 1, isolated[SemanticKey.Other], avg=True)

        if im_logging_enabled(data, LogLevel.Images):
            log_image(data, "floor_markers", 255. * floor_prob)
            log_image(data, "other_markers", 255. * other_prob)
            log_image(data, "ceiling_markers", 255. * ceiling_prob)
            log_image(data, "wall_like_markers", 255. * wall_like_prob)
            log_image(data, "wall_markers", 255. * wall_prob)

        segmentation_initial = np.int32(np.argmax(np.dstack(
            (.05 * np.ones_like(isolated[SemanticKey.Other]), other_prob, floor_prob, wall_prob, ceiling_prob, wall_like_prob)), -1))

        segmentation_initial[np.logical_and(segmentation_initial == 3, wall_prob < .5)] = 6
        m = np.logical_and(segmentation_initial == 2, other_prob > .5)
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

        log_segmentation_image(data, "segmentation_initial", segmentation_initial, img_lr)

        return segmentation_initial

    # Create fan from vertical vp
    def fan_surfaces(self, data, img_lr, locations, vp0, sure_walls, wall_mask, normals_c):

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

        log_segmentation_image(data, "fan1", labels_fan, img_lr)

        unique_labels = np.unique(labels_fan[labels_fan > 0])

        fan_normals_reduced = np.float32([np.mean(normals_c[labels_fan == j], 0) for j in unique_labels])
        lengths = np.sqrt(np.sum(fan_normals_reduced * fan_normals_reduced, -1))
        fan_normals_reduced /= np.dstack((lengths, lengths, lengths))[0]

        labels_fan, fan_normals_reduced, normals_wall = merge_by_angle_sweep(labels_fan, normals_wall,
                                                                             fan_normals_reduced, wall_mask > 0,
                                                                             angle_threshold=.8)
        log_segmentation_image(data, "fan2", labels_fan, img_lr)

        return labels_fan, fan_normals_reduced, normals_wall


    def run(self, data):

        img = data["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[1] > 1024:
            img = cv2.resize(img, (1024, int(img.shape[0] / img.shape[1] * 1024)))

        log_image(data, "image", img)

        output = np.float32(data["semantic_probs"])

        hed = data["hed"]
        log_image(data, 'hed', hed)

        h, w = output[0].shape

        shape = (w, h)

        hed_lr = cv2.resize(hed, shape)
        img_lr = cv2.resize(img, shape)

        sx = w / img.shape[1]
        sy = h / img.shape[0]

        #Include other types as part of floor: rug, earth, grass:
        self._combine_floor_masks(output)

        #break masks into major groups: Floor, Wall, Ceiling, etc
        isolated = self._isolate_masks(data, output)
        
        #get lines
        line_data, lines = find_lines(img, cv2.resize(hed, (img.shape[1], img.shape[0])), data["normals"])

        #get labeled lines image
        all_lines = self._get_lines_image(data, img_lr, lines, sx, sy)

        if im_logging_enabled(data, LogLevel.Segmentation):
            merged_lines = np.int32(np.zeros((img_lr.shape[0], img_lr.shape[1])))
            Line.draw_all(line_data, merged_lines, color=255, thickness=2, sx=sx, sy=sy, lineType=cv2.LINE_4)
            l_image_rgb = img_lr.copy()
            l_image_rgb[merged_lines > 0] = 255
            log_segmentation_image(data, "l_image", all_lines, img_lr)
            log_image(data, "l_image_rgb", l_image_rgb)

        # planes_data = data["planes"]
        # plane_parameters = np.array(data["planes"]["detection"][:, 6:9], dtype=np.float32)
        # plane_offsets = np.linalg.norm(plane_parameters, axis=-1, keepdims=True)
        # plane_normals = plane_parameters / np.maximum(plane_offsets, 1e-4)
        # plane_clusters = np.array(data["planes"]["detection"][:, 4], dtype=np.int32)
        # # roi = np.array(data["planes"]["detection"][:, 0:4], dtype=np.int32)
        # cluster_prob = data["planes"]["detection"][:, 5]

        # plane_masks = planes_data["masks"]
        # number_planes = len(plane_masks)
        # plane_rotations = np.zeros(number_planes, dtype=np.float32)

        # plane_XYZ = planes_data["plane_XYZ"][:, :, 80:-80, :].transpose(0, 2, 3, 1)

        # plane_masks = resize_array(plane_masks, shape)
        # plane_XYZ = resize_array(plane_XYZ, shape)

        # # depth = planes_data["depth_np"][:, 80:-80, :].transpose(1, 2, 0)
        # XYZ = planes_data["XYZ"][:, 80:-80, :].transpose(1, 2, 0)
        # # depth = cv2.resize(depth, shape)
        # XYZ = cv2.resize(XYZ, shape)            

        # normals = cv2.resize(data["normals"], shape)

        # if im_logging_enabled(data, LogLevel.Images):
        #     log_image(data, "XYZ", 255. * XYZ / np.amax(XYZ))
        #     log_image(data, "normals", normals)

        # normals = (normals - 127.5) / 127.5

        # floor_indices, ceiling_indices, horiz_indices, floor_angs = find_floor_indices(isolated[SemanticKey.Floor], isolated[SemanticKey.Ceiling],
        #                                                                                plane_masks, plane_normals)



        # # print("floor indices are: ", floor_indices, horiz_indices)
        # # print("ceiling indices are: ", ceiling_indices)

        # floor_normal = [0., 0, -1]
        # floor_index = -1

        # if len(floor_indices) > 0:
        #     floor_index = floor_indices[0]

        #     floor_normal = plane_normals[floor_index]
        #     floor_offset = plane_offsets[floor_index]
        # # print("floor_normal", floor_normal)

        # ceiling_index = -1

        # if len(ceiling_indices) > 0:
        #     ceiling_index = ceiling_indices[0]

        # wall_indices, vert_indices, vert_angs = find_wall_indices(isolated[SemanticKey.Wall], plane_masks, plane_normals[floor_index], plane_normals)

        # # print("wall indices are: ", wall_indices, vert_indices, vert_angs)

        # basis_indices = np.int32(np.concatenate([floor_indices, ceiling_indices, wall_indices]))

        # ceiling_normal = plane_normals[ceiling_index]
        # # print("ceiling normal, floor normal", ceiling_normal, floor_normal, np.dot(ceiling_normal,floor_normal))


        plane_geometry = PlaneGeometry(data, isolated, img_lr, shape)

        vert_indices = plane_geometry.dimensions[Dimension.Vertical].indices
        number_planes = len(plane_geometry.plane_masks)

        normals_combined, normals_nn_normals = combined_normals(-plane_geometry.normals, plane_geometry.plane_normals, plane_geometry.plane_masks, plane_geometry.basis_indices, plane_geometry.cluster_prob)
        normals_c = normals_combined

        lengths = np.maximum(np.sqrt(np.sum(normals_c * normals_c, -1)), 1e-6)
        normals_c /= np.dstack((lengths, lengths, lengths))

        if im_logging_enabled(data, LogLevel.Images):
            log_image(data, "normals_c_org", 127.5 * (normals_c + 1))

        cluster_masks = [.03 * np.ones_like(plane_geometry.plane_masks[0])]
        cluster_mask_indices = [0]
        for i in range(1, 8):
            clust = np.nonzero(plane_geometry.plane_clusters == i)[0]
            clust = np.intersect1d(clust, vert_indices)

            if len(clust) > 0:
                cluster_masks.append(np.sum(plane_geometry.plane_masks[clust], 0))
                cluster_mask_indices.append((i - 1) % 3 + 1)

        plane_cluster_seg = np.argmax(cluster_masks, 0)
        cluster_mask_indices = np.int32(cluster_mask_indices)
        plane_cluster_seg_rs = np.int32(
            cv2.resize(np.uint8(plane_cluster_seg), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST))

        log_segmentation_image(data, "plane_cluster_seg", plane_cluster_seg, img_lr)

        

        ######################################## Initial refinement work
        sure_walls = segmentation_initial = self._refine_surfaces(data, isolated, line_data, img_lr, hed_lr, sx, sy)

        edgelets = compute_edgelets(lines)

        vps, inliers, floor_normal, floor_offset, floor_rotation, fov, img_dir = compute_normal_from_vps(edgelets, img,
                                                                                                         data["fov"],
                                                                                                         plane_geometry.floor_normal,
                                                                                                         plane_geometry.floor_offset,
                                                                                                         isolated[SemanticKey.Floor])

        data["fov"] = fov

        #refine plane geometry, need to be in class, but getting there:
        if plane_geometry.floor_index > -1:
            plane_geometry.plane_parameters[plane_geometry.floor_index] = floor_normal * floor_offset

        log_image(data, "img_dir", img_dir)

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

        labels_fan, fan_normals_reduced, normals_wall = self.fan_surfaces(data, img_lr, locations, vp0, sure_walls, isolated[SemanticKey.Wall], normals_c)

        vl_image = np.zeros_like(all_lines)
        vl_image[sure_walls == 0] = 0

        ade_seg_c = np.dstack(
            (.95 * np.ones_like(isolated[SemanticKey.Other]), isolated[SemanticKey.Other], isolated[SemanticKey.Floor], isolated[SemanticKey.Wall], isolated[SemanticKey.Ceiling], isolated[SemanticKey.WallLike]))
        ade_seg = np.argmax(ade_seg_c, -1)

        if im_logging_enabled(data, LogLevel.Segmentation):
            
            log_image(data, "normals_wall_org", 127.5 * (normals_wall + 1))
            log_segmentation_image(data, "vl_image", vl_image, img_lr)
            log_segmentation_image(data, "ade_seg", np.int32(ade_seg), img_lr)

        plane_classes = get_planes_class(plane_geometry.plane_masks, ade_seg)
        # print("plane_classes", plane_classes)

        full_planes = plane_geometry.plane_masks.copy()
        full_planes[full_planes < .01] = 0

        plane_geometry.plane_masks[plane_geometry.plane_masks < .5] = 0
        wall_like_planes = [np.zeros_like(hed_lr / 255.)]

        wall_like_indices = []
        floor_indices = []
        ceiling_indices = []

        for k in range(number_planes):
            # print("cluster", plane_geometry.cluster_prob[k])
            a = np.int32(plane_classes[k])

            name = "plane_" + str(k)
            name += "_" + str(a)

            if a == 3 or a == 5:
                if k not in wall_like_indices:
                    wall_like_indices.append(k)
                    if plane_geometry.cluster_prob[k] > .5:
                        wall_like_planes.append(plane_geometry.plane_masks[k])
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

        log_segmentation_image(data, "wall_planes_seg", wall_planes_seg, img_lr)

        labels_wall_1 = labels_fan
        labels_wall_1[labels_wall_1 > 0] += np.amax(wall_planes_seg) + 1
        log_segmentation_image(data, "labels_wall_graph", labels_fan, img_lr, avg=False)

        for i in np.unique(labels_wall_1):
            if i == 0: continue

            mask = i == labels_wall_1
            m = mode(wall_planes_seg[np.logical_and(wall_planes_seg > 0, mask)])

            if len(m[0]) > 1:
                labels_wall_1[mask] = m[0]

        log_segmentation_image(data, "labels_wall_merge1", labels_wall_1, img_lr, avg=False)

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
                    plane_center = np.mean(plane_geometry.XYZ[label_mask], axis=0)
                    offset = np.dot(plane_center, plane_geometry.plane_normals[vert_not_wall[vert_arg]])
                    # print(all_vertical[wall_index], plane_offsets[all_vertical[wall_index]])
                    plane_geometry.plane_parameters[vert_not_wall[vert_arg]] = plane_geometry.plane_normals[vert_not_wall[vert_arg]] * offset
                else:
                    label_normal = np.mean(normals_wall[label_mask], 0)

                    label_normal /= max(np.linalg.norm(label_normal), .00001)
                    if len(all_vertical) > 0:
                        wall_index = np.argmax(np.dot(plane_geometry.plane_normals[all_vertical], label_normal))

                        plane_center = np.mean(plane_geometry.XYZ[label_mask], axis=0)
                        offset = np.dot(plane_center, label_normal)

                        plane_geometry.plane_parameters[all_vertical[wall_index]] = plane_geometry.plane_normals[all_vertical[wall_index]] * offset

                        labels_arg[label_mask] = all_vertical[wall_index] + 1
                        print("if no good match just take the closest by angle", all_vertical,
                              plane_geometry.plane_parameters[all_vertical[wall_index]], all_vertical[wall_index])

        plane_XYZ, plane_depth = calcPlaneXYZ(plane_geometry.plane_parameters, width=w, height=h, camera=camera, max_depth=10)
        log_segmentation_image(data, 'labels_arg', labels_arg, img_lr)


        #END planegeometry replacement

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

        log_segmentation_image(data, "labels_wall_merge2", labels_wall_2, img_lr, avg=False)

        final_masks = []

        final_plane_parameters = []
        final_rotations = []
        final_plane_XYZ = []

        wall_areas = []
        plane_rotations = np.zeros(number_planes, dtype=np.float32)

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
                plane_parameter[6:9] = plane_geometry.plane_parameters[l]
                plane_parameter[9] = 2
                plane_parameter[10] = plane_rotations[l]

                # log_image(data, str(l) + "plane_masks", 255. * plane_masks[l])

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
            isolated[SemanticKey.Floor] = np.uint8(segmentation_initial == 2)
            # floor_mask = cv2.dilate(floor_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            final_masks.append(255 * isolated[SemanticKey.Floor])

            plane_parameter = np.zeros((11))
            plane_parameter[:9] = data["planes"]["detection"][floor_indices[0]][:9]
            plane_parameter[6:9] = plane_geometry.plane_parameters[floor_indices[0]]
            plane_parameter[9] = 1
            plane_parameter[10] = plane_rotations[floor_indices[0]]

            final_plane_parameters.append(plane_parameter)
            final_plane_XYZ.append(plane_XYZ[floor_indices[0]])
            final_rotations.append(plane_rotations[floor_indices[0]])

        # add ceiling
        if len(ceiling_indices) > 0:
            isolated[SemanticKey.Ceiling] = 255 * np.uint8(segmentation_initial == 4)
            final_masks.append(isolated[SemanticKey.Ceiling])
            plane_parameter = np.zeros((11))
            plane_parameter[:9] = data["planes"]["detection"][ceiling_indices[0]][:9]
            plane_parameter[6:9] = plane_geometry.plane_parameters[ceiling_indices[0]]
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
        log_image(data, 'lighting_smooth', lighting_smooth)

        data["lighting"] = lighting_smooth

        log_image(data, 'lighting', lighting_smooth)

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

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_segmentation_image(data, "pre_final_labels", final_labels_hr - 1, img)

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

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_segmentation_image(data, "final_labels", np.int32(final_labels) - 1, img)




