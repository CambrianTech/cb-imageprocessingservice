from pipeline.input import PipelineInput

import os
try:
    from imageio import imread
except:
    from scipy.misc import imread

import pickle
from pathlib import Path

class PipelineFileInput(PipelineInput):
    def __init__(self, base_path):
        super().__init__(base_path)

    def run(self, data):

        path = Path(data["path"])

        if path.suffix == ".pickle":
            print("Reading data from", path)
            with open(path, 'rb') as handle:
                loaded = pickle.load(handle)
                #todo: maybe there's a deep copy that works instead? 
                for key in loaded:
                    data[key] = loaded[key]
        else:
            print("Reading image", data["path"])
            data["image"] = imread(data["path"])
            data["image"] = data["image"][:, :, :3]