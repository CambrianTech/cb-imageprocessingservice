from io import BytesIO
import boto3
import numpy as np
import cv2
from pipeline.core import PipelineStep


def _get_image_from_s3(s3_client, bucket: str, key: str) -> np.ndarray:
    response = s3_client.Object(bucket, key).get()
    image_data = response["Body"].read()
    image_arr = np.fromstring(image_data, np.uint8)
    return cv2.imdecode(image_arr, cv2.CV_LOAD_IMAGE_COLOR)


class PipelineGetData(PipelineStep):
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")

    @property
    def required_keys(self) -> list:
        return ["image_s3_key"]

    @property
    def output_keys(self) -> list:
        return ["image"]

    def run(self, data):
        data["image"] = _get_image_from_s3(
            self.s3_client, self.bucket_name, data["image_s3_key"])
