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

from pipeline.core import PipelineStep

@abstract
class PipelineFileOutput(PipelineStep):
    def __init__(self, base_path):
        super().__init__(base_path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)

    @property
    def required_keys(self) -> list:
        return ["semantic", "lighting", "superpixels"]

    @property
    def output_keys(self) -> list:
        return ["semantic_url", "lighting_url", "data_url" "superpixels_url"]

    @abstractmethod
    def save_image(image, filename):
        imsave(os.path.join(self.base_path, filename), image)
