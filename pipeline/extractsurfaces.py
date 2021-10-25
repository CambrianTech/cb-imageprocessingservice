from enum import Enum
import numpy as np
import cv2

from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .logging import im_logging_enabled, log_image, LogLevel, log_segmentation_image
from .ade20k import ADE20K
from .semanticlabel import SemanticLabel

floor_like = [ADE20K.earth, ADE20K.grass, ADE20K.rug]
wall_like = [ADE20K.windowpane, ADE20K.door, ADE20K.curtain, ADE20K.mirror, ADE20K.painting, ADE20K.shelf, ADE20K.column, ADE20K.screen_door, ADE20K.blind, ADE20K.projection_screen]
ceiling_like = [ADE20K.light]
box_like = [ADE20K.cabinet, ADE20K.dishwasher, ADE20K.oven, ADE20K.fireplace, ADE20K.kitchen]

def isolate_masks(data, output):

    isolated = list([None] * (SurfaceType.max_index() + 1))

    isolated[SurfaceType.Floor] = output[ADE20K.floor.index].copy()
    isolated[SurfaceType.Wall] = output[ADE20K.wall.index].copy()
    isolated[SurfaceType.Ceiling] = output[ADE20K.ceiling.index].copy()
    
    def combine_outputs(grouping, labels):
        isolated[grouping] = np.zeros_like(isolated[SurfaceType.Floor])
        for label in labels:
            isolated[grouping] += output[label.index]

    #group wall like, floor like, ceiling like
    combine_outputs(SurfaceType.FloorLike, floor_like)
    combine_outputs(SurfaceType.WallLike, wall_like)
    combine_outputs(SurfaceType.CeilingLike, ceiling_like)

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

        #Consolidate types: Include other types as part of floor: rug, earth, grass
        output = np.float32(data["semantic_probs"])
        data["output"] = output

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

        
