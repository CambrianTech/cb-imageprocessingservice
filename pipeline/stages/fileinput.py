from .input import PipelineInput

import os
try:
    from imageio import imread
except:
    from scipy.misc import imread

import pickle
from pathlib import Path

from termcolor import colored

class PipelineFileInput(PipelineInput):

    def run(self, data):

        path = Path(data["path"])

        if path.suffix == ".pickle":
            print("Reading data from", colored(path, 'cyan', attrs=['bold']))
            with open(path, 'rb') as handle:
                loaded = pickle.load(handle)
                #todo: maybe there's a deep copy that works instead? 
                for key in loaded:
                    data[key] = loaded[key]
        else:
            print("Reading image", data["path"])
            data["image"] = imread(data["path"])
            data["image"] = data["image"][:, :, :3]