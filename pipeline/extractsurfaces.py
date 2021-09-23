from enum import Enum
import numpy as np
import cv2

from .core import PipelineStep, PipelineStepIndex
from .logging import im_logging_enabled, log_image, LogLevel, log_segmentation_image
from .ade20k import ADE20K
from .semanticlabel import SemanticLabel

floor_like = [ADE20K.earth, ADE20K.grass, ADE20K.rug, ADE20K.light]
wall_like = [ADE20K.windowpane, ADE20K.door, ADE20K.curtain, ADE20K.mirror, ADE20K.painting, ADE20K.shelf, ADE20K.column, ADE20K.screen_door, ADE20K.blind, ADE20K.projection_screen]
ceiling_like = [ADE20K.fan, ADE20K.chandelier]

#Keep major (floor, wall, ceiling) even, "Like" versions odd. 
#Perhaps write class method
class Groupings(SemanticLabel):
    Floor=0
    FloorLike=1

    Wall=2
    WallLike=3

    Ceiling=4
    CeilingLike=5

    Other=6

class SurfaceTypes(SemanticLabel):
    Floor=0
    Wall=1
    Ceiling=2
    Other=3

def isolate_masks(data, output):

    isolated = list([None] * (Groupings.max_index() + 1))

    isolated[Groupings.Floor] = output[ADE20K.floor.index].copy()
    isolated[Groupings.Wall] = output[ADE20K.wall.index].copy()
    isolated[Groupings.Ceiling] = output[ADE20K.ceiling.index].copy()
    
    def combine_outputs(grouping, labels):
        isolated[grouping] = np.zeros_like(isolated[Groupings.Floor])
        for label in labels:
            isolated[grouping] += output[label.index]

    #group wall like, floor like, ceiling like
    combine_outputs(Groupings.FloorLike, floor_like)
    combine_outputs(Groupings.WallLike, wall_like)
    combine_outputs(Groupings.CeilingLike, ceiling_like)

    #label everything else as other
    isolated[Groupings.Other] = 1.0 - sum(isolated[:-1])

    surface_types = list([None] * (SurfaceTypes.max_index() + 1))

    surface_types[SurfaceTypes.Floor] = isolated[Groupings.Floor] + isolated[Groupings.FloorLike]
    surface_types[SurfaceTypes.Wall] = isolated[Groupings.Wall] + isolated[Groupings.WallLike]
    surface_types[SurfaceTypes.Ceiling] = isolated[Groupings.Ceiling] + isolated[Groupings.CeilingLike]
    surface_types[SurfaceTypes.Other] = isolated[Groupings.Other]

    return isolated, surface_types

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

        log_image(data, "image", data["image"])

        if data["image"].shape[0] > shape[0] or data["image"].shape[1] > shape[1]:
            data["downscaled"] = cv2.resize(data["image"], shape)
        else:
            data["downscaled"] = data["image"]

        if im_logging_enabled(data, LogLevel.Segmentation):
            probs = np.dstack((tuple(output)))
            log_segmentation_image(data, "segmentation", np.argmax(probs, -1), data["downscaled"])

        #combine_floor_masks(output)
        data["isolated"], data["surface_types"] = isolate_masks(data, output) #break masks into major groups: Floor, Wall, Ceiling, etc

        if im_logging_enabled(data, LogLevel.Segmentation):
            isolated_probs = np.dstack(data["isolated"])
            log_segmentation_image(data, "isolated", np.argmax(isolated_probs, -1), data["downscaled"], labelset=Groupings)

            surface_probs = np.dstack(data["surface_types"])
            log_segmentation_image(data, "surface_probs", np.argmax(surface_probs, -1), data["downscaled"], labelset=SurfaceTypes)

        
