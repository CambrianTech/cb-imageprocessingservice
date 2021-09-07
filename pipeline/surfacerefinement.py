import numpy as np
from scipy import ndimage
import cv2
from skimage.morphology import skeletonize, remove_small_objects
from skimage.segmentation import join_segmentations, watershed

import cambrian.image_processing as ip

from .core import PipelineStep, PipelineStepIndex
from .Line import Line
from .extractsurfaces import Groupings
from .logging import get_segmentation_image, log_segmentation_image, im_logging_enabled, log_image, LogLevel

class SurfaceRefinement():
    def __init__(self, image, hed, probs, lines):
        super().__init__()
        self.image = image
        self.hed = hed
        self.probs = probs
        self.lines = lines
        
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

    def refine(self, data, confidence=0.95):
        #get labeled lines image
        sx = self.image.shape[0] / data["image"].shape[0]
        sy = self.image.shape[1] / data["image"].shape[1]

        items = self.probs.copy()
        items.insert(0, (confidence * np.ones_like(self.probs[Groupings.Other])))

        ade_seg_c = np.dstack(tuple(items))
        ade_seg = np.argmax(ade_seg_c, -1)

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_segmentation_image(data, "probs", np.int32(ade_seg), self.image, show_legend=False)



        #contours, hierarchy = cv2.findContours(np.uint8(final_labels == d + 2), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        
        


        #draw lines in BW
        merged_lines = np.int32(np.zeros((self.image.shape[0], self.image.shape[1])))
        Line.draw_all(merged_lines, self.lines, color=255, thickness=2, sx=sx, sy=sy)

        watershed_mask = (merged_lines == 0)

        other_markers = np.int32(
            self._refine_surface(self.probs[Groupings.Other], self.image, big_thresh=.001, small_thresh=.95, watershed_dist=.03, gradient=False))

        wall_markers = np.int32(
            self._refine_surface(self.probs[Groupings.Wall], self.hed, big_thresh=.05, small_thresh=.95, watershed_dist=.05, gradient=True, watershed_mask=watershed_mask))

        floor_markers = np.int32(
            self._refine_surface(self.probs[Groupings.Floor], self.image, big_thresh=.001, small_thresh=.95, watershed_dist=.05, gradient=False))

        wall_like_markers = np.int32(
            self._refine_surface(self.probs[Groupings.WallLike], self.image, big_thresh=.001, small_thresh=.95, watershed_dist=.05, gradient=False))

        ceiling_markers = np.int32(
            self._refine_surface(self.probs[Groupings.Ceiling], self.hed, big_thresh=.05, small_thresh=.95, watershed_dist=.05, gradient=True, watershed_mask=watershed_mask))

        ceiling_prob = get_segmentation_image(ceiling_markers + 1, self.probs[Groupings.Ceiling], avg=True)
        wall_like_prob = get_segmentation_image(wall_like_markers + 1, self.probs[Groupings.WallLike], avg=True)
        wall_like_prob[wall_like_prob < .25] = 0
        wall_prob = get_segmentation_image(wall_markers + 1, self.probs[Groupings.Wall], avg=True)

        ade_skel = skeletonize(self.probs[Groupings.Other] > .5)
        self.probs[Groupings.Floor][ade_skel > 0] = 0
        floor_prob = get_segmentation_image(floor_markers + 1, self.probs[Groupings.Floor], avg=True)
        floor_prob[floor_prob < .25] = 0
        floor_markers[floor_prob < .25] = 100
        other_markers = join_segmentations(floor_markers, other_markers)

        self.probs[Groupings.Other][ade_skel > 0] = 1
        other_prob = get_segmentation_image(other_markers + 1, self.probs[Groupings.Other], avg=True)

        if im_logging_enabled(data, LogLevel.Images):
            log_image(data, "floor_markers", 255. * floor_prob)
            log_image(data, "other_markers", 255. * other_prob)
            log_image(data, "ceiling_markers", 255. * ceiling_prob)
            log_image(data, "wall_like_markers", 255. * wall_like_prob)
            log_image(data, "wall_markers", 255. * wall_prob)

        segmentation = np.int32(np.argmax(np.dstack(
            (.05 * np.ones_like(self.probs[Groupings.Other]), other_prob, floor_prob, wall_prob, ceiling_prob, wall_like_prob)), -1))

        segmentation[np.logical_and(segmentation == 3, wall_prob < .5)] = 6
        m = np.logical_and(segmentation == 2, other_prob > .5)
        segmentation[merged_lines > 0] = 0
        segmentation[m] = 1


        for i in range(1, 7):
            pruned = remove_small_objects(segmentation == i, min_size=32)
            segmentation[np.logical_and(segmentation == i, pruned == 0)] = 0

        line_mask = np.zeros((segmentation.shape[0], segmentation.shape[1]))
        Line.draw_all(line_mask, self.lines, color=255, thickness=2, sx=sx, sy=sy)

        segmentation = watershed(self.hed, segmentation,
                                         mask=line_mask == 0)
        segmentation = cv2.watershed(self.image, segmentation)

        segmentation[line_mask > 0] = 0
        distances = cv2.distanceTransform(np.uint8(line_mask), cv2.DIST_L1, 3)

        distances = np.uint8(distances)
        segmentation = cv2.watershed(cv2.cvtColor(np.uint8(distances), cv2.COLOR_GRAY2BGR),
                                             segmentation)

        log_segmentation_image(data, "segmentation_refined", segmentation, self.image)


        return segmentation


class PipelineSurfaceRefinement(PipelineStep):

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.SurfaceRefinement

    @property
    def required_keys(self) -> list:
        return ["downscaled", "isolated", "lines", "hed"]

    @property
    def output_keys(self) -> list:
        return ["segmentation"]

    def run(self, data):

        img_lr = data["downscaled"]
        shape = (img_lr.shape[1], img_lr.shape[0])

        log_image(data, 'hed', data["hed"])
        hed_lr = cv2.resize(data["hed"], shape)

        refiner = SurfaceRefinement(img_lr, hed_lr, data["isolated"], data["lines"])
        data["segmentation"] = refiner.refine(data)

