from io import BytesIO
import boto3
import cv2
import numpy as np
from pipeline.core import PipelineStep


def _upload_image_to_s3(s3_client, image: np.ndarray, bucket: str, key: str):
    success, buffer = cv2.imencode(".png", image)

    if not success:
        raise ValueError("cv2.imencode not successful, invalid image?")

    image_data = BytesIO(buffer)
    s3_client.upload_fileobj(image_data, bucket, key)


class PipelineUploadResults(PipelineStep):
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")

    @property
    def required_keys(self) -> list:
        return ["semantic", "lighting"]

    @property
    def output_keys(self) -> list:
        return ["semantic_url", "lighting_url"]

    def run(self, data):
        key_semantic = "%s_semantic.png" % data["image_s3_key"]
        key_lighting = "%s_lighting.png" % data["image_s3_key"]

        _upload_image_to_s3(
            self.s3_client, data["semantic_probs"][:, :, 0], self.bucket_name, key_semantic)
        _upload_image_to_s3(
            self.s3_client, data["lighting"], self.bucket_name, key_lighting)

        # TODO: Get URLs in a better way
        data["semantic_url"] = "https://s3.amazonaws.com/%s/%s" % (self.bucket_name, key_semantic)
        data["lighting_url"] = "https://s3.amazonaws.com/%s/%s" % (self.bucket_name, key_lighting)