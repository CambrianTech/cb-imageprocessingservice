from enum import Enum
import numpy as np
import cv2

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import im_logging_enabled, log_image, LogLevel, log_segmentation_image
from pipeline.data.ade20k import ADE20K
from pipeline.data.semanticlabel import SemanticLabel

on_floor = [ADE20K.earth, ADE20K.grass, ADE20K.rug]
on_wall = [ADE20K.windowpane, ADE20K.door, ADE20K.curtain, ADE20K.mirror, ADE20K.painting, ADE20K.shelf, ADE20K.column, ADE20K.screen_door, ADE20K.blind, ADE20K.projection_screen]
on_ceiling = [ADE20K.light]
box_like = [ADE20K.cabinet, ADE20K.dishwasher, ADE20K.oven, ADE20K.fireplace, ADE20K.kitchen]
legged_objects = [ADE20K.table, ADE20K.chair, ADE20K.bed, ADE20K.cabinet, ADE20K.chest, ADE20K.coffee_table, ADE20K.stool, ADE20K.bench, ADE20K.ottoman, ADE20K.armchair, ADE20K.chest]

def isolate_masks(data, output):

    isolated = list([None] * (SurfaceType.max_index() + 1))

    isolated[SurfaceType.Floor] = output[ADE20K.floor.index]
    isolated[SurfaceType.Wall] = output[ADE20K.wall.index]
    isolated[SurfaceType.Ceiling] = output[ADE20K.ceiling.index]
    
    def combine_outputs(grouping, labels):
        isolated[grouping] = np.zeros_like(isolated[SurfaceType.Floor])
        for label in labels:
            isolated[grouping] += output[label.index]

    #group wall like, floor like, ceiling like
    combine_outputs(SurfaceType.OnFloor, on_floor)
    combine_outputs(SurfaceType.OnWall, on_wall)
    combine_outputs(SurfaceType.OnCeiling, on_ceiling)

    #label everything else as other
    isolated[SurfaceType.Other] = 1.0 - sum(isolated[:-1])

    return isolated

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

    def run(self, data):

        data["lighting"] = cv2.edgePreservingFilter(np.uint8(data["lighting"]), flags=1, sigma_s=10, sigma_r=1.0)
        log_image(data, 'lighting_smooth', data["lighting"])

        #Consolidate types: Include other types as part of floor: rug, earth, grass
        output = data["semantic_probs"]

        h, w = output[0].shape
        shape = (w, h)

        log_image(data, "hed", data["hed"])

        if data["image"].shape[0] > shape[0] or data["image"].shape[1] > shape[1]:
            data["downscaled"] = cv2.resize(data["image"], shape)
        else:
            data["downscaled"] = data["image"]

        #combine_floor_masks(output)
        data["isolated"] = isolate_masks(data, output) #break masks into surface types

        if im_logging_enabled(data, LogLevel.Segmentation):
            isolated_probs = np.dstack(data["isolated"])
            log_segmentation_image(data, "surface_probs", np.argmax(isolated_probs, -1), data["downscaled"], labelset=SurfaceType)

            probs = np.dstack(output)
            log_segmentation_image(data, "everything", np.argmax(probs, -1), data["downscaled"])

        
