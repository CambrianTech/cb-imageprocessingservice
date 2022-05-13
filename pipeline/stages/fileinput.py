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

    def get(self, info: dict) -> dict:

        path = Path(info["path"])
        print("Reading data from", colored(path, 'cyan', attrs=['bold']))

        if path.suffix == ".pickle":
            with open(path, 'rb') as handle:
                return pickle.load(handle)
        else:
            data = dict()
            data["image"] = imread(str(path))
            data["image"] = data["image"][:, :, :3]
            return data
