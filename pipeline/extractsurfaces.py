from enum import Enum
import numpy as np
import cv2

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.logging import im_logging_enabled, log_image, LogLevel, log_segmentation_image
from pipeline.ade20k import ADE20K
from pipeline.semanticlabel import SemanticLabel

floor_like = [ADE20K.earth, ADE20K.grass, ADE20K.rug]
wall_like = [ADE20K.windowpane, ADE20K.door, ADE20K.curtain, ADE20K.painting, ADE20K.shelf, ADE20K.column, ADE20K.screen_door, ADE20K.blind, ADE20K.projection_screen]
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

#Exported from https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo = 150.csv 
def combine_floor_masks(output):
        
    output[ADE20K.floor.index] += output[ADE20K.rug.index]
    output[ADE20K.rug.index] = 0

    output[ADE20K.floor.index] += output[ADE20K.earth.index]
    output[ADE20K.earth.index] = 0

    output[ADE20K.floor.index] += output[ADE20K.grass.index]
    output[ADE20K.grass.index] = 0


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
    isolated[Groupings.Other] = 1.0 - isolated[Groupings.Floor] - isolated[Groupings.Wall] - isolated[Groupings.WallLike] - isolated[Groupings.Ceiling]

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

        if data["image"].shape[0] > shape[0] or data["image"].shape[1] > shape[1]:
            data["downscaled"] = cv2.resize(data["image"], shape)
        else:
            data["downscaled"] = data["image"]

        if im_logging_enabled(data, LogLevel.Segmentation):
            probs = np.dstack((tuple(output)))
            log_segmentation_image(data, "segmentation_raw", np.argmax(probs, -1), data["downscaled"])

        #combine_floor_masks(output)
        isolated_masks = isolate_masks(data, output) #break masks into major groups: Floor, Wall, Ceiling, etc

        if im_logging_enabled(data, LogLevel.Segmentation):
            isolated_probs = np.dstack(isolated_masks)
            log_segmentation_image(data, "segmentation_isolated", np.argmax(isolated_probs, -1), data["downscaled"], labelset=Groupings)

        data["isolated"] = isolated_masks

        

        
