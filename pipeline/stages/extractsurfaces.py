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
from pipeline.misc.utils import list_flatten
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

    def refine_semantics(self, labels, lines, label_freedoms:list, name="semantic"):

        image = self.data["downscaled"]
        markers = np.zeros(image.shape[:2], dtype=np.int32)

        watershed_image = cv2.resize(self.data["hed"], (image.shape[1], image.shape[0]))
        watershed_mask = np.zeros(markers.shape, dtype=np.int32)

        min_matches = 100

        all_labels = np.unique(labels).astype(np.int32)

        semantic_key = {}

        for label in all_labels:

            label_mask = labels == label

            value = int(label + 1)

            match = next(filter(lambda x: x.index_of(value) >= 0, label_freedoms), None)

            freedom = match.freedom if match is not None else None

            if freedom is not None and len(labels[label_mask]) > min_matches:

                index = match.index_of(value)
                semantic_label = match.labels[index]

                semantic_key[value] = semantic_label.name

                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask[label_mask] = 1
                watershed_mask[label_mask] = 1

                mask_padded = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
                contours, _ = cv2.findContours(mask_padded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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


        #log_mask(self.data, "%s_mask" % name, watershed_mask)
        #log_segmentation_image(self.data, "%s_markers" % name, markers, self.data["downscaled"], labelset=semantic_key)

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

            labels[markers_mask] = label

        #log_segmentation_image(self.data, "%s_refined" % name, markers, self.data["downscaled"], labelset=semantic_key)


    def run(self, data):
        self.data = data
        self.image = self.data["downscaled"]

        diagonal = math.hypot(self.image.shape[0], self.image.shape[1])

        line_candidates = self.data["vp_lines"]

        #Consolidate types: Include other types as part of floor: rug, earth, grass        
        if im_logging_enabled(data):
            log_segmentation_image(self.data, "semantic_labels_raw", self.data["semantic_labels"], self.data["downscaled"])

        self.refine_semantics(self.data["semantic_labels"], line_candidates, [ LF([ADE20K.ceiling], 0.2), LF([ADE20K.wall], 0.2), LF(box_like, 0.03), LF(on_wall, 0.1)], name="barriers_wc")
        self.refine_semantics(self.data["semantic_labels"], line_candidates, [ LF([ADE20K.wall], 0.02), LF(box_like, 0.03), LF([ADE20K.floor], 0.05), \
                              LF([ADE20K.stairs, ADE20K.stairway], 0.05), LF(on_floor, 0.05), LF(legged_objects, 0.05)], name="barriers_floor")

        #combine_floor_masks(output)
        self.data["isolated_probs"], self.data["isolated_labels"] = isolate_masks(data, self.data["semantic_probs"], self.data["semantic_labels"]) #break masks into surface types


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

                epsilon = 3
                polygon = cv2.approxPolyDP(contour, epsilon, True)
                num_pts = len(polygon)

                for i in range(num_pts):
                    point_a = polygon[i][0]
                    point_b = polygon[(i+1) % num_pts][0]

                    length = distance.euclidean(point_a, point_b)

                    if length >= min_line_length and not line_on_image_edge(point_a, point_b, self.image.shape[1], self.image.shape[0], min_distance=3):
                        new_lines.append(Line(point_a[0], point_a[1], point_b[0], point_b[1], group="semantic_lines"))

        new_lines = list(filter(lambda l: line_within_mask(l, data["vp_mask"]), new_lines))

        valid_vps = [data["vertical_vp"]] + data["horizontal_vps"]

        intersections = []

        filtered_lines = []
        for vp in valid_vps:
            inliers = list(get_inliers(new_lines, vp.model, np.radians(8)))
            filtered_lines.extend(inliers)

            vp.inliers = np.concatenate((vp.inliers, inliers), axis=0)

        filtered_lines = merge_lines(filtered_lines, search_width=max(diagonal/200, 3), search_length=1.05, angle_threshold=np.radians(10))

        self.data["semantic_lines"] = filtered_lines
        self.data["vp_lines"].extend(filtered_lines)
        
        # combined = self.data["lines"] + self.data["semantic_lines"]
        # #merge_lines(combined, search_width=max(diagonal/100, 3), remove_matches=False)

        # clusters = list(set(map(lambda x: x.cluster, self.data["semantic_lines"])))

        # for line in [line for line in self.data["lines"] if line.cluster in clusters]:
        #     line.group = "semantic_lines"

        #self.data["semantic_lines"] = list(filter(lambda x: x.group == "semantic_lines", combined))

        #self.data["semantic_lines"], intersections = extend_to_intersection(self.data["semantic_lines"], search_length=1.5, search_width=3.0, min_angle_difference=np.radians(15))

        #self.data["lines"] = merge_lines(self.data["lines"] + self.data["semantic_lines"], search_width=max(diagonal/400, 3))

        #filter out after merge, matching passed in group
        #self.data["semantic_lines"] = list(filter(lambda x: x.group == "semantic_lines", self.data["lines"]))

        #self.data["lines"] = merge_lines(self.data["lines"], search_width=max(diagonal/150, 3), remove_matches=False)

        # clusters = list(set(map(lambda x: x.cluster, self.data["semantic_lines"])))
        
        # for line in [line for line in self.data["lines"] if line.cluster in clusters]:
        #     line.group = "semantic_lines"

        #self.data["semantic_lines"] = list(filter(lambda x: x.group == "semantic_lines", self.data["lines"]))

        #self.data["vp_lines"], intersections = extend_to_intersection(self.data["vp_lines"], search_length=1.1, search_width=3.0, min_angle_difference=0)

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

            
