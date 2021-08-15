from io import BytesIO
import boto3
import numpy as np
from pipeline.core import PipelineStep
import os
try:
    from imageio import imread
except:
    from scipy.misc import imread

import pickle
from pathlib import Path

def _get_image_from_s3(s3_client, bucket: str, key: str) -> np.ndarray:
    data = BytesIO()
    s3_client.download_fileobj(bucket, "%s/background" % key, data)
    data.seek(0)
    return imread(data)  # uint8 [0, 255]


class PipelineBucketSource(PipelineStep):
    def __init__(self, bucket_name=None):
        super().__init__()
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")

    @property
    def required_keys(self) -> list:
        return ["unique_id"]

    @property
    def output_keys(self) -> list:
        return ["image"]

    def run(self, data):
        # Get image from S3 or local folder if local dir is set.
        if self.local_directory is None:
            data["image"] = _get_image_from_s3(
                self.s3_client, self.bucket_name, data["unique_id"])
        else:
            local_path = os.path.join(
                self.local_directory, data["unique_id"])
            data["image"] = imread(local_path)

        data["image"] = data["image"][:, :, :3]

class PipelineFileSource(PipelineStep):
    def __init__(self):
        super().__init__()

    @property
    def required_keys(self) -> list:
        return ["path"]

    @property
    def output_keys(self) -> list:
        return ["image"]

    def run(self, data):

        path = Path(data["path"])

        if path.suffix == ".pickle":
            print("Reading data from", path)
            with open(path, 'rb') as handle:
                loaded = pickle.load(handle)
                #todo: maybe there's a deep copy that works instead? 
                for key in loaded:
                    data[key] = loaded[key]
        else:
            print("Reading image", data["path"])
            data["image"] = imread(data["path"])
            data["image"] = data["image"][:, :, :3]

