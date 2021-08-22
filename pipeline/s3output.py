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

from pipeline.output import PipelineOutput

class PipelineS3Output(PipelineOutput):
    def __init__(self, base_path, s3Client:S3Client):
        super().__init__(base_path, s3_url="https://s3.amazonaws.com")
        self.s3_client = s3Client
        self.s3_url = s3_url

    @property
    def required_keys(self) -> list:
        return ["semantic", "lighting", "superpixels"]

    @property
    def output_keys(self) -> list:
        return ["semantic_url", "lighting_url", "data_url" "superpixels_url"]

     @protected
    def make_url(self, path):
        return "%s/%s" % (self.s3_url, path)

    @abstractmethod
    def save_image(image, filename):
        self.s3_client.upload_image_to_s3(image, self.base_path, filename)
        