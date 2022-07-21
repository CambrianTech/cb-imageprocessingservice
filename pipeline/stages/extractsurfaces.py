from enum import Enum
import numpy as np
import cv2
from skimage.segmentation import watershed
import itertools
import math
from scipy.spatial import distance

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_image, im_logging_enabled, log_segmentation_image, log_mask, LogLevel
from pipeline.data.ade20k import ADE20K, floor, on_floor, wall, on_wall, ceiling, on_ceiling, legged_objects, box_like
from pipeline.data.semanticlabel import SemanticLabel
from pipeline.misc.utils import list_flatten, resize_array
from pipeline.components.line import Line, extend_to_intersection, draw_lines, line_within_mask, line_on_image_edge, merge_lines
from pipeline.stages.vanishingpointfinder import get_inliers

class LF():

    def __init__(self, labels, freedom):
        self.labels = list_flatten(labels)
        self.freedom = freedom

    def index_of(self, label):
        try:
            return self.labels.index(label)
        except:
            return -1

def isolate_masks(data, semantic_probs, semantic_labels):

    isolated_probs = list([None] * (SurfaceType.max_index() + 1))
    isolated_labels = -1 * np.ones_like(semantic_labels)
    
    def combine_outputs(grouping, label_list):
        isolated_probs[grouping] = np.zeros_like(semantic_probs[ADE20K.floor.index])

        for label in label_list:
            isolated_probs[grouping.index] += semantic_probs[label.index]
            isolated_labels[semantic_labels == label.index] = grouping.index


    #group wall like, floor like, ceiling like
    combine_outputs(SurfaceType.Floor, floor)
    combine_outputs(SurfaceType.OnFloor, on_floor)

    combine_outputs(SurfaceType.Wall, wall)
    combine_outputs(SurfaceType.OnWall, on_wall)

    combine_outputs(SurfaceType.Ceiling, ceiling)
    combine_outputs(SurfaceType.OnCeiling, on_ceiling)

    #label everything else as other
    isolated_probs[SurfaceType.Other] = 1.0 - sum(isolated_probs[:-1])
    isolated_labels[isolated_labels < 0] = SurfaceType.Other.index

    return isolated_probs, isolated_labels

class PipelineExtractSurfaces(PipelineStep):

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.ExtractSurfaces

    @property
    def required_keys(self) -> list:
        return ["downscaled", "semantic_probs"]

    @property
    def output_keys(self) -> list:
        return ["semantic_labels", "isolated_probs", "isolated_labels"]

    def refine_semantics(self, segmentation, label_freedoms:list=None, mask=None, freedom=None):

        lines = self.data["vp_lines"]
        image = self.data["downscaled"]
        markers = np.zeros(image.shape[:2], dtype=np.int32)

        watershed_image = cv2.resize(self.data["hed"], (image.shape[1], image.shape[0]))
        watershed_mask = np.zeros(markers.shape, dtype=np.int32)

        min_matches = 100

        all_labels = np.unique(segmentation).astype(np.int32)

        for label in all_labels:

            label_mask = segmentation == label

            if mask is not None:
                label_mask = np.where(label_mask & (mask > 0))

            value = int(label + 1)

            if label_freedoms is not None:
                match = next(filter(lambda x: x.index_of(value) >= 0, label_freedoms), None)
                freedom = match.freedom if match is not None else None

            if freedom is not None and len(segmentation[label_mask]) > min_matches:

                seg_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                seg_mask[label_mask] = 1
                watershed_mask[label_mask] = 1

                seg_mask = cv2.copyMakeBorder(seg_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                contours, _ = cv2.findContours(seg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    shape = contour.shape
                    contour = (contour.flatten() - 1).reshape(shape)

                    sub_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(sub_mask, [contour], -1, 1, thickness=cv2.FILLED)
                    cv2.drawContours(markers, [contour], -1, value, thickness=cv2.FILLED)

                    thickness = int(np.sqrt(cv2.contourArea(contour)) / 15)
                    cv2.drawContours(markers, [contour], -1, 0, thickness=thickness)

                    # # #padded transform
                    mask_padded = cv2.copyMakeBorder(sub_mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                    dist_transform = cv2.distanceTransform(mask_padded, cv2.DIST_L2, 5)
                    dist_transform = dist_transform[1:-1,1:-1]

                    markers[dist_transform > freedom * dist_transform.max()] = value

        line_candidates = list(filter(lambda line: line_within_mask(line, markers, value=0), lines))

        draw_lines(watershed_mask, line_candidates, color=0, thickness=1, lineType=cv2.LINE_4)

        markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))
        markers[markers < 0] = 0
        markers[watershed_mask == 0] = 0

        #remove lines, gaps
        markers = markers.astype(np.uint8)
        markers = cv2.morphologyEx(markers, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

        for value in np.unique(markers):

            if value == 0: continue
            
            label = value - 1
            markers_mask = markers == value

            segmentation[markers_mask] = label

        

    def run(self, data):
        self.data = data
        self.image = self.data["downscaled"]

        diagonal = math.hypot(self.image.shape[0], self.image.shape[1])


        #Consolidate types: Include other types as part of floor: rug, earth, grass        
        if im_logging_enabled(data):
            log_segmentation_image(self.data, "semantic_labels_raw", self.data["semantic_labels"], self.data["downscaled"])

        self.refine_semantics(self.data["semantic_labels"], label_freedoms=[ LF([ADE20K.ceiling], 0.2), LF([ADE20K.wall], 0.2), LF(box_like, 0.03), LF(on_wall, 0.1)])

        self.refine_semantics(self.data["semantic_labels"], label_freedoms=[ LF([ADE20K.wall], 0.02), LF(box_like, 0.03), LF([ADE20K.floor], 0.05), \
                                LF([ADE20K.stairs, ADE20K.stairway], 0.05), LF(on_floor, 0.05), LF(legged_objects, 0.05)])

        #combine_floor_masks(output)
        self.data["isolated_probs"], self.data["isolated_labels"] = isolate_masks(data, self.data["semantic_probs"], self.data["semantic_labels"]) #break masks into surface types

        # shape = (self.image.shape[1], self.image.shape[0])
        # probs = resize_array(self.data["planes"]["masks"], shape)

        # index_mask = np.dstack(tuple(probs))
        # index_mask = np.int32(np.argmax(index_mask, -1))

        # self.refine_semantics(index_mask, freedom=0.1)

        # log_segmentation_image(self.data, "index_mask", index_mask, self.data["downscaled"])

        # data["index_mask"] = index_mask

        #now pull lines and add them to vp_lines and lines
        new_lines = []
        min_line_length = diagonal / 60

        for surfaceType in SurfaceType:
            mask = np.zeros( self.image.shape[:2], dtype=np.uint8)
            mask[self.data["isolated_labels"] == surfaceType] = 1

            mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0) 
            _contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in _contours:
                shape = contour.shape
                contour = (contour.flatten() - 1).reshape(shape)
                contour_length = cv2.arcLength(contour, True)

                epsilon = 1.5
                polygon = cv2.approxPolyDP(contour, epsilon, True)
                num_pts = len(polygon)

                for i in range(num_pts):
                    point_a = polygon[i][0]
                    point_b = polygon[(i+1) % num_pts][0]

                    length = distance.euclidean(point_a, point_b)

                    if length >= min_line_length and not line_on_image_edge(point_a, point_b, self.image.shape[1], self.image.shape[0], min_distance=3):
                        new_lines.append(Line(point_a[0], point_a[1], point_b[0], point_b[1], group="semantic_lines"))

        new_lines = list(filter(lambda l: line_within_mask(l, data["vp_mask"]), new_lines))
        new_lines = merge_lines(new_lines, search_width=max(diagonal/200, 3), search_length=1.05, angle_threshold=np.radians(10))

        valid_vps = [data["vertical_vp"]] + data["horizontal_vps"]

        intersections = []

        filtered_lines = []
        for vp in valid_vps:
            if len(new_lines) < 1: break

            inliers = list(get_inliers(new_lines, vp.model, np.radians(8)))
            
            vp.inliers = np.concatenate((vp.inliers, inliers), axis=0)
            filtered_lines.extend(inliers)

            #get remaining
            new_lines = list(set(new_lines).difference(set(inliers)))


        self.data["semantic_lines"] = filtered_lines
        self.data["vp_lines"].extend(filtered_lines)
        

        if im_logging_enabled(data):
            lines_image =  self.image.copy()

            draw_lines(lines_image, self.data["vp_lines"])
            draw_lines(lines_image, self.data["semantic_lines"], color=(255,150,0), thickness=2, lineType=cv2.LINE_AA)

            for point in intersections:
                cv2.circle(lines_image, (int(point[0]), int(point[1])), 5, (255, 255, 0), cv2.FILLED, cv2.LINE_AA)

            log_image(self.data, "semantic_lines", lines_image)

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_segmentation_image(self.data, "semantic_labels", self.data["semantic_labels"], self.data["downscaled"])

            isolated_probs = np.argmax(np.dstack(self.data["isolated_probs"]), -1)
            log_segmentation_image(self.data, "isolated_probs", isolated_probs, self.data["downscaled"], labelset=SurfaceType)

            log_segmentation_image(self.data, "isolated_labels", self.data["isolated_labels"], self.data["downscaled"], labelset=SurfaceType)

            
