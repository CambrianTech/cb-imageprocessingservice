from enum import Enum
import numpy as np
import cv2
from skimage.segmentation import watershed
import itertools

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_image, im_logging_enabled, log_segmentation_image, log_mask, LogLevel
from pipeline.data.ade20k import ADE20K
from pipeline.data.semanticlabel import SemanticLabel
from pipeline.misc.utils import list_flatten
from pipeline.components.line import extend_lines, draw_lines

floor = [ADE20K.floor, ADE20K.grass, ADE20K.earth, ADE20K.sidewalk]
on_floor = [ADE20K.rug]

wall = [ADE20K.wall]
on_wall = [ADE20K.windowpane, ADE20K.door, ADE20K.curtain, ADE20K.mirror, ADE20K.painting, ADE20K.shelf, ADE20K.column, ADE20K.screen_door, ADE20K.blind, ADE20K.projection_screen, ADE20K.radiator, ADE20K.sconce, ADE20K.towel]

ceiling = [ADE20K.ceiling]
on_ceiling = [ADE20K.light, ADE20K.chandelier]

lights = [ADE20K.light, ADE20K.lamp]
box_like = [ADE20K.cabinet, ADE20K.dishwasher, ADE20K.oven, ADE20K.fireplace, ADE20K.kitchen]
legged_objects = [ADE20K.table, ADE20K.chair, ADE20K.bed, ADE20K.cabinet, ADE20K.chest, ADE20K.coffee_table, ADE20K.stool, ADE20K.bench, ADE20K.ottoman, ADE20K.armchair, ADE20K.chest]

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
        return ["image", "semantic_probs"]

    @property
    def output_keys(self) -> list:
        return ["output", "isolated"]

    def refine_semantics(self, labels, label_freedoms:list, name="semantic"):

        image = self.data["downscaled"]
        markers = np.zeros(image.shape[:2], dtype=np.int32)

        watershed_image = cv2.resize(self.data["hed"], (image.shape[1], image.shape[0]))
        watershed_mask = np.zeros(markers.shape, dtype=np.int32)

        min_matches = 100
        num_labels = np.amax(labels) + 1

        semantic_key = {}

        for label in range(0, num_labels):

            label_mask = labels == label

            value = label + 1

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


        def is_barrier_line(line, mask, num_points=7, num_matches=3):
            line_points = np.linspace(line.point_a, line.point_b, num_points)
            count = 0
            for point in line_points:
                if point[0] < 0 or point[0] >= mask.shape[1] or point[1] < 0 or point[1] >= mask.shape[0]: continue

                if mask[int(point[1]), int(point[0])] == 0:
                    count += 1

                if count > num_matches:
                    return True

            return False
        
        line_candidates = list(filter(lambda line: is_barrier_line(line, markers), self.data["vp_lines"]))
        line_candidates = extend_lines(line_candidates)

        draw_lines(watershed_mask, line_candidates, color=0, thickness=1, lineType=cv2.LINE_4)

        #log_mask(self.data, "%s_mask" % name, watershed_mask)
        #log_segmentation_image(self.data, "%s_markers" % name, markers, self.data["downscaled"], labelset=semantic_key)

        markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))
        markers[markers<0] = 0
        markers[watershed_mask == 0] = 0

        for value in np.unique(markers):

            if value == 0: continue
            
            label = value - 1
            markers_mask = markers == value
            labels[markers_mask] = label

        #log_segmentation_image(self.data, "%s_refined" % name, markers, self.data["downscaled"], labelset=semantic_key)


    def run(self, data):
        self.data = data

        #Consolidate types: Include other types as part of floor: rug, earth, grass
        self.data["semantic_labels"] = np.argmax(np.dstack(self.data["semantic_probs"]), -1)

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_segmentation_image(self.data, "semantic_labels_raw", self.data["semantic_labels"], self.data["downscaled"])

        self.refine_semantics(self.data["semantic_labels"], [ LF([ADE20K.ceiling], 0.2), LF([ADE20K.wall], 0.2), LF(box_like, 0.03), LF(on_wall, 0.1)], name="barriers_wc")
        self.refine_semantics(self.data["semantic_labels"], [ LF([ADE20K.wall], 0.02), LF(box_like, 0.03), LF([ADE20K.floor], 0.05), \
                              LF([ADE20K.stairs, ADE20K.stairway], 0.05), LF(on_floor, 0.05), LF(legged_objects, 0.05)], name="barriers_floor")

        #combine_floor_masks(output)
        self.data["isolated_probs"], self.data["isolated_labels"] = isolate_masks(data, self.data["semantic_probs"], self.data["semantic_labels"]) #break masks into surface types

        if im_logging_enabled(data, LogLevel.Segmentation):
            log_segmentation_image(self.data, "semantic_labels", self.data["semantic_labels"], self.data["downscaled"])

            isolated_probs = np.argmax(np.dstack(self.data["isolated_probs"]), -1)
            log_segmentation_image(self.data, "isolated_probs", isolated_probs, self.data["downscaled"], labelset=SurfaceType)

            log_segmentation_image(self.data, "isolated_labels", self.data["isolated_labels"], self.data["downscaled"], labelset=SurfaceType)

            
