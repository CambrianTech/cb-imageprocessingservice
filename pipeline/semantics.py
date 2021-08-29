from enum import Enum
import numpy as np

from pipeline.logging import im_logging_enabled, log_image, LogLevel
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