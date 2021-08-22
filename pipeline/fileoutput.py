from io import BytesIO
try:
    from imageio import imsave
except:
    from scipy.misc import imsave
import boto3
import numpy as np
import os
import json
import zlib
from pathlib import Path

from pipeline.output import PipelineOutput

class PipelineFileOutput(PipelineOutput):
    def __init__(self, base_path):
        super().__init__(base_path)

    def save_image(self, image, filename, url):
        path = Path(os.path.join(self.base_path, url))
        if len(path.parents) > 0:
            path.parents[0].mkdir(parents=True, exist_ok=True)

        imsave(path, image)

    def save_data(self, data, filename, url):
        path = Path(os.path.join(self.base_path, url))
        if len(path.parents) > 0:
            path.parents[0].mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as out_file:
            json.dump(data, out_file, indent=4)
