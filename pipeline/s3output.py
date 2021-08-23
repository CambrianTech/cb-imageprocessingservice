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

from pipeline.s3client import S3Client
from pipeline.output import PipelineOutput

class PipelineS3Output(PipelineOutput):
    def __init__(self, base_path, s3Client:S3Client):
        super().__init__(base_path, s3_url="https://s3.amazonaws.com")
        self.s3_client = s3Client
        self.s3_url = s3_url

    def make_url(self, path):
        return "%s/%s" % (self.s3_url, path)

    def save_image(self, image, filename, url):
        self.s3_client.upload_image_to_s3(image, self.base_path, filename)

    def save_data(self, data, filename, url):
        self.s3_client.upload_json_to_s3(data, self.base_path, filename)
        