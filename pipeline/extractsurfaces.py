from enum import Enum
import numpy as np
import cv2

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.logging import im_logging_enabled, log_image, LogLevel, log_segmentation_image
from pipeline.ade20k import ADE20K, wall_like

class Groupings(Enum):
    Wall = "wall"
    Floor = "floor"
    Ceiling = "ceiling"
    WallLike = "wall-like"
    Other = "other"

#Exported from https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo = 150.csv 
def combine_floor_masks(output):
        
    output[ADE20K.floor.index] += output[ADE20K.rug.index]
    output[ADE20K.rug.index] = 0

    output[ADE20K.floor.index] += output[ADE20K.earth.index]
    output[ADE20K.earth.index] = 0

    output[ADE20K.floor.index] += output[ADE20K.grass.index]
    output[ADE20K.grass.index] = 0

def isolate_masks(data, output):

    isolated = {}

    isolated[Groupings.Wall] = output[ADE20K.wall.index].copy()
    isolated[Groupings.Floor] = output[ADE20K.floor.index].copy()
    isolated[Groupings.Ceiling] = output[ADE20K.ceiling.index].copy()
    isolated[Groupings.WallLike] = np.zeros_like(isolated[Groupings.Wall])

    for label in wall_like:
        isolated[Groupings.WallLike] += output[label.index]

    isolated[Groupings.Other] = 1.0 - isolated[Groupings.Floor] - isolated[Groupings.Wall] - isolated[Groupings.WallLike] - isolated[Groupings.Ceiling]

    if im_logging_enabled(data, LogLevel.Segmentation):
        for key in isolated.keys():
            log_image(data, key.value, 255. * isolated[key])

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

        combine_floor_masks(output)
        isolated_masks = isolate_masks(data, output) #break masks into major groups: Floor, Wall, Ceiling, etc

        if im_logging_enabled(data, LogLevel.Segmentation):
            isolated_probs = np.dstack((isolated_masks[Groupings.Other], isolated_masks[Groupings.Floor], isolated_masks[Groupings.Wall], isolated_masks[Groupings.Ceiling], isolated_masks[Groupings.WallLike]))
            log_segmentation_image(data, "segmentation_isolated", np.argmax(isolated_probs, -1), data["downscaled"])

        data["isolated"] = isolated_masks

        

        
