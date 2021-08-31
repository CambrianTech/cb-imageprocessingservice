import cv2
import numpy as np
import random
from scipy import ndimage
from scipy.stats import mode
from skimage.morphology import skeletonize
from skimage.segmentation import watershed

from pipeline.core import PipelineStep
from pipeline.logging import get_segmentation_image, log_image, log_segmentation_image, log_ply, im_logging_enabled, LogLevel
from pipeline.ade20k import ADE20K
from pipeline.extractsurfaces import Groupings
from cambrian.VanishingPointFinder import VanishingPointFinder
from pipeline.poseestimator import calcPlaneXYZ, fan_surfaces
from pipeline.planegeometry import Dimension

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

class PipelineMergeSurfaces(PipelineStep):

    def __init__(self):
        super().__init__()

    @property
    def required_keys(self) -> list:
        return ["image", "output", "lines", "isolated", "segmentation"]

    @property
    def output_keys(self) -> list:
        return ["mask", "lighting"]

    def run(self, data):
        self.img = data["image"]

        self.output = data["output"]

        self.height, self.width = self.img.shape[:2]
        self.diagonal = np.hypot(self.width, self.height)

        h, w = self.output[0].shape

        #finalize angles
        isolated = data["isolated"]
        img_lr = data["downscaled"]
        segmentation_initial = data["segmentation"]
        sure_walls = (segmentation_initial == 3)

        camera = data["camera"]
        plane_masks = data["plane_masks"]
        plane_normals = data["plane_normals"]
        plane_parameters = data["plane_parameters"]
        cluster_prob = data["cluster_prob"]
        XYZ = data["xyz"]
        hed_lr = cv2.resize(data["hed"], (img_lr.shape[1], img_lr.shape[0]))
        vert_indices = data["dimensions"][Dimension.Vertical].indices


        number_planes = len(plane_masks)

        labels_fan, fan_normals_reduced, normals_wall = fan_surfaces(data, img_lr, data["edgelets"][0], data["vp0"], sure_walls, isolated[Groupings.Wall], data["normals_c"])

        
        vl_image = np.int32(np.zeros((img_lr.shape[0], img_lr.shape[1])))
        vl_image[sure_walls == 0] = 0

        ade_seg_c = np.dstack(
            (.95 * np.ones_like(isolated[Groupings.Other]), isolated[Groupings.Other], isolated[Groupings.Floor], isolated[Groupings.Wall], isolated[Groupings.Ceiling], isolated[Groupings.WallLike]))
        ade_seg = np.argmax(ade_seg_c, -1)

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_image(data, "normals_wall_org", 127.5 * (normals_wall + 1))
            log_segmentation_image(data, "vl_image", vl_image, img_lr)
            log_segmentation_image(data, "ade_seg", np.int32(ade_seg), img_lr)

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
