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
        return ["image_s3_key"]

    @property
    def output_keys(self) -> list:
        return ["image"]

    def run(self, data):
        session = boto3.session.Session()
        s3_client = session.client('s3')

        # Get image from S3 or local folder if local dir is set.
        if "image_local_dir" not in data:
            data["image"] = _get_image_from_s3(
                s3_client, self.bucket_name, data["image_s3_key"])
        else:
            local_path = os.path.join(
                data["image_local_dir"], self.bucket_name, data["image_s3_key"])
            data["image"] = imread(local_path)

        data["image"] = data["image"][:, :, :3]

        

        if data["image"].shape[0] > shape[0] or data["image"].shape[1] > shape[1]:
            data["downscaled"] = cv2.resize(data["image"], shape)
        else:
            data["downscaled"] = data["image"]
