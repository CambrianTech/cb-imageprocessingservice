import numpy as np
from scipy import ndimage
import cv2

import cambrian.image_processing as ip
from cambrian.Line import Line

from pipeline.extractsurfaces import Groupings
from pipeline.logging import get_segmentation_image, log_segmentation_image, im_logging_enabled, log_image, LogLevel
from skimage.morphology import skeletonize, remove_small_objects
from skimage.segmentation import join_segmentations, watershed

class SurfaceRefinement():
    def __init__(self, image, hed, masks, line_data, lines):
        super().__init__()
        self.image = image
        self.hed = hed
        self.masks = masks
        self.line_data = line_data
        self.lines = lines

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
        

    def _refine_surface(self, mask, image, big_thresh=.03, small_thresh=.97, watershed_dist=.05, watershed_mask=None, gradient=True):
        small = ip.refine_mask_watershed(None, image, np.uint8(mask > small_thresh), None, distance=watershed_dist, gradient=gradient,
                                       watershed_mask=watershed_mask)

        big = ip.refine_mask_watershed(None, image, np.uint8(mask > big_thresh), None, distance=watershed_dist, gradient=gradient,
                                         watershed_mask=watershed_mask)

        markers = np.dstack((np.ones_like(big), small, big))
        markers = np.argmax(markers, -1)

        markers[markers == 1] = (ndimage.label(markers == 1)[0])[markers == 1] + np.amax(markers)
        markers[markers == 2] = (ndimage.label(markers == 2)[0])[markers == 2] + np.amax(markers)
        return markers

    def refine(self, data):
        #get labeled lines image
        sx = self.image.shape[0] / data["image"].shape[0]
        sy = self.image.shape[1] / data["image"].shape[1]

        all_lines = self._get_lines_image(data, self.image, self.lines, sx, sy)

        #draw lines in BW
        merged_lines = np.int32(np.zeros((self.image.shape[0], self.image.shape[1])))
        Line.draw_all(self.line_data, merged_lines, color=255, thickness=2, sx=sx, sy=sy, lineType=cv2.LINE_4)

        if im_logging_enabled(data, LogLevel.Segmentation):
            l_image_rgb = self.image.copy()
            l_image_rgb[merged_lines > 0] = 255
            log_segmentation_image(data, "l_image", all_lines, self.image)
            log_image(data, "l_image_rgb", l_image_rgb)

        watershed_mask = (merged_lines == 0)

        other_markers = np.int32(
            self._refine_surface(self.masks[Groupings.Other], self.image, big_thresh=.001, small_thresh=.95, watershed_dist=.03, gradient=False))

        wall_markers = np.int32(
            self._refine_surface(self.masks[Groupings.Wall], self.hed, big_thresh=.05, small_thresh=.95, watershed_dist=.05, gradient=True, watershed_mask=watershed_mask))

        floor_markers = np.int32(
            self._refine_surface(self.masks[Groupings.Floor], self.image, big_thresh=.001, small_thresh=.95, watershed_dist=.05, gradient=False))

        wall_like_markers = np.int32(
            self._refine_surface(self.masks[Groupings.WallLike], self.image, big_thresh=.001, small_thresh=.95, watershed_dist=.05, gradient=False))

        ceiling_markers = np.int32(
            self._refine_surface(self.masks[Groupings.Ceiling], self.hed, big_thresh=.05, small_thresh=.95, watershed_dist=.05, gradient=True, watershed_mask=watershed_mask))

        ceiling_prob = get_segmentation_image(ceiling_markers + 1, self.masks[Groupings.Ceiling], avg=True)
        wall_like_prob = get_segmentation_image(wall_like_markers + 1, self.masks[Groupings.WallLike], avg=True)
        wall_like_prob[wall_like_prob < .25] = 0
        wall_prob = get_segmentation_image(wall_markers + 1, self.masks[Groupings.Wall], avg=True)

        ade_skel = skeletonize(self.masks[Groupings.Other] > .5)
        self.masks[Groupings.Floor][ade_skel > 0] = 0
        floor_prob = get_segmentation_image(floor_markers + 1, self.masks[Groupings.Floor], avg=True)
        floor_prob[floor_prob < .25] = 0
        floor_markers[floor_prob < .25] = 100
        other_markers = join_segmentations(floor_markers, other_markers)

        self.masks[Groupings.Other][ade_skel > 0] = 1
        other_prob = get_segmentation_image(other_markers + 1, self.masks[Groupings.Other], avg=True)

        if im_logging_enabled(data, LogLevel.Images):
            log_image(data, "floor_markers", 255. * floor_prob)
            log_image(data, "other_markers", 255. * other_prob)
            log_image(data, "ceiling_markers", 255. * ceiling_prob)
            log_image(data, "wall_like_markers", 255. * wall_like_prob)
            log_image(data, "wall_markers", 255. * wall_prob)

        segmentation = np.int32(np.argmax(np.dstack(
            (.05 * np.ones_like(self.masks[Groupings.Other]), other_prob, floor_prob, wall_prob, ceiling_prob, wall_like_prob)), -1))

        segmentation[np.logical_and(segmentation == 3, wall_prob < .5)] = 6
        m = np.logical_and(segmentation == 2, other_prob > .5)
        segmentation[merged_lines > 0] = 0
        segmentation[m] = 1


        for i in range(1, 7):
            pruned = remove_small_objects(segmentation == i, min_size=32)
            segmentation[np.logical_and(segmentation == i, pruned == 0)] = 0

        line_mask = Line.draw_all(self.line_data,
                                  np.zeros((segmentation.shape[0], segmentation.shape[1])),
                                  color=255,
                                  thickness=2, sx=sx, sy=sy, lineType=cv2.LINE_4)

        segmentation = watershed(self.hed, segmentation,
                                         mask=line_mask == 0)
        segmentation = cv2.watershed(self.image, segmentation)

        segmentation[line_mask > 0] = 0
        distances = cv2.distanceTransform(np.uint8(line_mask), cv2.DIST_L1, 3)

        distances = np.uint8(distances)
        segmentation = cv2.watershed(cv2.cvtColor(np.uint8(distances), cv2.COLOR_GRAY2BGR),
                                             segmentation)

        log_segmentation_image(data, "segmentation_initial", segmentation, self.image)


        return segmentation
