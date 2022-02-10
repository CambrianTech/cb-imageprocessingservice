from io import BytesIO
import boto3.session
import numpy as np
from pipeline.core import PipelineStep
import os
try:
    from imageio import imread
except:
    from scipy.misc import imread


def _get_image_from_s3(s3_client, bucket: str, key: str) -> np.ndarray:
    data = BytesIO()
    s3_client.download_fileobj(bucket, "%s/background" % key, data)
    data.seek(0)
    return imread(data)  # uint8 [0, 255]


class PipelineGetData(PipelineStep):
    def __init__(self, bucket_name):
        super().__init__()
        self.bucket_name = bucket_name

    @property
    def required_keys(self) -> list:
        return ["unique_id"]

    @property
    def output_keys(self) -> list:
        return ["image"]

    def run(self, data):
        session = boto3.session.Session()
        s3_client = session.client('s3')

        # Get image from S3 or local folder if local dir is set.
        if "image_local_dir" not in data:
            data["image"] = _get_image_from_s3(
                s3_client, self.bucket_name, data["unique_id"])
        else:
            local_path = os.path.join(
                data["image_local_dir"], self.bucket_name, data["unique_id"])
            data["image"] = imread(local_path)

        data["image"] = data["image"][:, :, :3]
