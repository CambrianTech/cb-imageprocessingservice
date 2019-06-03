from io import BytesIO
try:
    from imageio import imsave
except:
    from scipy.misc import imsave
import boto3.docs.method
import numpy as np
import os.path
from pipeline.core import PipelineStep


def _upload_image_to_s3(s3_client, image: np.ndarray, bucket: str, key: str):
    if image.dtype == np.float32:
        image = (255 * image).astype(np.uint8)

    image_data = BytesIO()
    imsave(image_data, image, format=".png")
    image_data.seek(0)
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

        mask_image = data["mask"]
        lighting_image = data["lighting"]

        # Upload to S3 or write to local folder if local dir is set.
        if "results_local_dir" not in data:
            _upload_image_to_s3(
                self.s3_client, mask_image, self.bucket_name, key_semantic)
            _upload_image_to_s3(
                self.s3_client, lighting_image, self.bucket_name, key_lighting)

            # TODO: Get URLs in a better way
            data["semantic_url"] = "https://s3.amazonaws.com/%s/%s" % (
                self.bucket_name, key_semantic)
            data["lighting_url"] = "https://s3.amazonaws.com/%s/%s" % (
                self.bucket_name, key_lighting)
        else:
            mask_path = os.path.join(
                data["results_local_dir"], self.bucket_name, key_semantic)
            lighting_path = os.path.join(
                data["results_local_dir"], self.bucket_name, key_lighting)

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            os.makedirs(os.path.dirname(lighting_path), exist_ok=True)

            # Convert dtypes if necessary
            if mask_image.dtype == np.float32:
                mask_image = (255 * mask_image).astype(np.uint8)
            if lighting_image.dtype == np.float32:
                lighting_image = (255 * lighting_image).astype(np.uint8)

            imsave(mask_path, mask_image)
            imsave(lighting_path, lighting_image)

            data["semantic_url"] = "http://127.0.0.1:8080/getimage/%s/%s" % (
                self.bucket_name, key_semantic)
            data["lighting_url"] = "http://127.0.0.1:8080/getimage/%s/%s" % (
                self.bucket_name, key_lighting)
