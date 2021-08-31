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

    def save_image(self, image, filename, url, quality=None):
        self.pipeline.s3_client.upload_image_to_s3(image, self.pipeline.dest_path, os.path.join(self.unique_id, url))

    def save_file(self, data, filename, url):
        self.pipeline.s3_client.upload_bytes_to_s3(data, self.pipeline.dest_path, os.path.join(self.unique_id, url))

    def save_data(self, data, filename, url):
        self.pipeline.s3_client.upload_json_to_s3(data, self.pipeline.dest_path, os.path.join(self.unique_id, url))
        