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
from skimage.morphology import skeletonize, remove_small_objects

import pickle
import math
from skimage.segmentation import watershed
from scipy.stats import mode

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.surface_type import SurfaceType
from pipeline.misc.utils import resize_array, get_segmentation_image, calculate_plane_xyz
from .planegeometry import PlaneGeometry, Dimension
from .surfacerefinement import SurfaceRefinement
from pipeline.data.ade20k import ADE20K
from pipeline.data.logging import log_image, log_segmentation_image, im_logging_enabled, LogLevel
from .poseestimator import PoseEstimator, fan_surfaces

def rough_dilate_erode(is_dilate, mask, size=5, iterations=1, scale=0.5, maintain_size=True, interpolation=cv2.INTER_NEAREST):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    shape = mask.shape
    mask = cv2.resize(mask, (int(shape[1] * scale), int(shape[0] * scale)), interpolation)
    mask = cv2.dilate(mask, kernel, iterations=iterations) if is_dilate else cv2.erode(mask, kernel, iterations=iterations)
    if maintain_size:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation)
    return mask

def gabor_filter(bw, theta, lambd, gamma=0.0, psi=0.0):
    ksize = lambd
    sigma = ksize * lambd
    result = cv2.filter2D(bw, cv2.CV_8UC1, cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F))
    return result

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
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Refine

    @property
    def description(self) -> str:
        return super().description + " (legacy)"

    @property
    def required_keys(self) -> list:
        return ["image", "output", "hed", "mask", "isolated"]

    @property
    def output_keys(self) -> list:
        return ["planes", "mask", "lighting", "floor_rotation"]

    def transfer_labels(self, data, final_labels, final_plane_number, final_plane_parameters):
        data["planes"]["masks"] = np.zeros((final_plane_number, final_labels.shape[0], final_labels.shape[1]),
                                           dtype=np.uint8)

        data["planes"]["detection"] = np.zeros((final_plane_number, 11), dtype=data["planes"]["detection"].dtype)
        data["planes"]["detection"] = final_plane_parameters
        # data["planes"]["rotation"] = final_rotations

        for d in range(final_plane_number):
            contours, hierarchy = cv2.findContours(np.uint8(final_labels == d + 2), cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)

            data["planes"]["masks"][d] = np.zeros_like(np.uint8(final_labels == d + 2))

            if len(contours) > 0:

                for i in range(len(contours)):
                    area = cv2.contourArea(contours[i])

                    if area > 4 * 16 * 16:
                        if hierarchy[0, i, 3] == -1:  # this is the outer contour which we need to draw
                            cv2.drawContours(data["planes"]["masks"][d], [contours[i]], -1, 255, -1, cv2.LINE_AA)
                        else:
                            cv2.drawContours(data["planes"]["masks"][d], contours, i, 0, -1)

                data["planes"]["detection"][d, 0:4] = [0, 0, 0, 0]

    def run(self, data):

        img = data["image"]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if img.shape[1] > 1024:
            img = cv2.resize(img, (1024, int(img.shape[0] / img.shape[1] * 1024)))

        log_image(data, "image", img)

        output = data["output"]

        hed = data["hed"]
        log_image(data, 'hed', hed)

        h, w = output[0].shape

        shape = (w, h)

        hed_lr = cv2.resize(hed, shape)
        img_lr = cv2.resize(img, shape)

        sx = w / img.shape[1]
        sy = h / img.shape[0]

        #process lighting
        lighting_rgb = np.uint8(data["lighting"])
        lighting_smooth = cv2.edgePreservingFilter(lighting_rgb, flags=1, sigma_s=10, sigma_r=1.0)
        log_image(data, 'lighting_smooth', lighting_smooth)
        data["lighting"] = lighting_smooth
        log_image(data, 'lighting', lighting_smooth)

        #break masks into major groups: Floor, Wall, Ceiling, etc
        isolated = data["isolated"]
        
        plane_geometry = PlaneGeometry(data, isolated, img_lr)
        plane_geometry.process()

        vert_indices = plane_geometry.dimensions[Dimension.Vertical].indices
        number_planes = len(plane_geometry.plane_masks)

        ######################################## Initial refinement work
        refiner = SurfaceRefinement(data, isolated)
        segmentation_initial = refiner.refine(data)
        sure_walls = (segmentation_initial == ADE20K.floor.index) #shouldn't this be == ADE20K.wall.index

        pose_estimator = PoseEstimator(data, img, data["lines"], data["fov"], isolated[SurfaceType.Floor], plane_geometry.floor_normal, plane_geometry.floor_offset)
        pose_estimator.estimate()

        data["fov"] = pose_estimator.fov
        data["floor_rotation"] = pose_estimator.floor_rotation
        
        if plane_geometry.floor_index > -1:
            plane_geometry.plane_parameters[plane_geometry.floor_index] = pose_estimator.floor_normal * pose_estimator.floor_offset

        labels_fan, fan_normals_reduced, normals_wall = fan_surfaces(data, img_lr, pose_estimator.edgelets[0], pose_estimator.vp0, sure_walls, isolated[SurfaceType.Wall], plane_geometry.normals_c)

        vl_image = np.int32(np.zeros((img_lr.shape[0], img_lr.shape[1])))
        vl_image[sure_walls == 0] = 0

        ade_seg_c = np.dstack(
            (.95 * np.ones_like(isolated[SurfaceType.Other]), isolated[SurfaceType.Other], isolated[SurfaceType.Floor], isolated[SurfaceType.Wall], isolated[SurfaceType.Ceiling], isolated[SurfaceType.OnWall]))
        ade_seg = np.argmax(ade_seg_c, -1)

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_image(data, "normals_wall_org", 127.5 * (normals_wall + 1))
            log_segmentation_image(data, "ade_seg", np.int32(ade_seg), img_lr, labelset=ADE20K)

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

        plane_XYZ, plane_depth = calculate_plane_xyz(plane_geometry.plane_parameters, width=w, height=h, camera=pose_estimator.camera, max_depth=10)
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
            isolated[SurfaceType.Floor] = np.uint8(segmentation_initial == 2)
            # floor_mask = cv2.dilate(floor_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
            final_masks.append(255 * isolated[SurfaceType.Floor])

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
            isolated[SurfaceType.Ceiling] = 255 * np.uint8(segmentation_initial == 4)
            final_masks.append(isolated[SurfaceType.Ceiling])
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
        final_labels[vl_image > 0] = 0

        mask_res = 2048
        mask_shape = (mask_res, int(mask_res * img.shape[0] / img.shape[1]))

        if img.shape[0] > img.shape[1]:
            mask_shape = (int(mask_res * img.shape[1] / img.shape[0]), mask_res)

        final_masks_hr = resize_array(np.uint8(final_masks), mask_shape)
        final_labels_hr = np.int32(np.argmax(final_masks_hr, 0))

        final_labels_hr[final_labels_hr > 0] += 1
        final_labels_hr[final_masks_hr[0] > 0] = 1
        final_labels_hr += 1

        final_merged_lines = np.zeros_like(final_labels_hr)

        #def draw_lines(img, lines, color=(255,50,255,255), thickness=1, sx=1.0, sy=1.0, lineType=cv2.LINE_8):
        draw_lines(final_merged_lines, data["lines"], color=1, thickness=6, 
                   sx=final_labels_hr.shape[1] / img.shape[1], sy=final_labels_hr.shape[0] / img.shape[0], lineType=cv2.LINE_AA)

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
            pruned = remove_small_objects(mask, 100)  # pruned[inter > 0] = 1
            final_labels_hr[mask > 0] = 0
            final_labels_hr[pruned > 0] = i

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_segmentation_image(data, "pre_final_labels", final_labels_hr - 1, img)

        final_labels_hr = cv2.watershed(cv2.cvtColor(distances, cv2.COLOR_GRAY2BGR), np.int32(final_labels_hr))
        final_labels_hr[final_labels_hr < 0] = 0
        final_labels = np.int32(final_labels_hr)

        self.transfer_labels(data, final_labels, final_plane_number, final_plane_parameters)

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_segmentation_image(data, "final_labels", np.int32(final_labels) - 1, img)


