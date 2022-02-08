import os

from .s3client import S3Client
from pipeline.stages.output import PipelineOutput

class PipelineS3Output(PipelineOutput):

    def save_image(self, image, filename, url, quality=None):
        self.pipeline.s3_client.upload_image_to_s3(image, self.config.dest_path, os.path.join(self.unique_id, url))

    def save_file(self, data, filename, url):
        self.pipeline.s3_client.upload_bytes_to_s3(data, self.config.dest_path, os.path.join(self.unique_id, url))

    def save_data(self, data, filename, url):
        self.pipeline.s3_client.upload_json_to_s3(data, self.config.dest_path, os.path.join(self.unique_id, url))
        