from io import BytesIO
try:
    from imageio import imsave
except:
    from scipy.misc import imsave
import boto3.docs.method
import numpy as np
import os.path
import json

from pipeline.core import PipelineStep


def _upload_image_to_s3(s3_client, image: np.ndarray, bucket: str, key: str):
    if image.dtype == np.float32:
        image = (255 * image).astype(np.uint8)

    image_data = BytesIO()
    imsave(image_data, image, format=".png")
    image_data.seek(0)
    s3_client.upload_fileobj(image_data, bucket, key)

def _upload_json_to_s3(s3_client, json_dict: dict, bucket: str, key: str):
    json_data = BytesIO()
    json_data.write(json.dumps(json_dict, indent=5).encode())
    json_data.seek(0)
    s3_client.upload_fileobj(json_data, bucket, key)


class PipelineUploadResults(PipelineStep):
    def __init__(self, bucket_name):
        self.bucket_name = bucket_name
#        self.bucket_name = "hart-develop-2vlai3b8"
        self.s3_client = boto3.client("s3")

    @property
    def required_keys(self) -> list:
        return ["semantic", "lighting", "superpixels"]

    @property
    def output_keys(self) -> list:
        return ["semantic_url", "lighting_url", "data_url" "superpixels_url"]

    def run(self, data):
        key_semantic = "%s_semantic.png" % data["image_s3_key"]
        key_lighting = "%s_lighting.png" % data["image_s3_key"]
        key_data = "%s_data.json" % data["image_s3_key"]
        key_superpixels = "%s_superpixels.png" % data["image_s3_key"]

        json_dict = {"cameraPosition": [0.0, data["camera_elevation"], 0.0], "cameraRotation": data["camera_rotation"], "floorRotation": data["floor_rotation"], "fov": data["fov"]}

        mask_image = data["mask"]
        lighting_image = data["lighting"]
        superpixels_image = data["superpixels"]

        # Upload to S3 or write to local folder if local dir is set.
        if "results_local_dir" not in data:
            _upload_image_to_s3(
                self.s3_client, mask_image, self.bucket_name, key_semantic)
            _upload_image_to_s3(
                self.s3_client, lighting_image, self.bucket_name, key_lighting)
            _upload_json_to_s3(
                self.s3_client, json_dict, self.bucket_name, key_data)
            _upload_image_to_s3(
                self.s3_client, superpixels_image, self.bucket_name, key_superpixels)

            # TODO: Get URLs in a better way
            data["semantic_url"] = "https://s3.amazonaws.com/%s/%s" % (
                self.bucket_name, key_semantic)
            data["lighting_url"] = "https://s3.amazonaws.com/%s/%s" % (
                self.bucket_name, key_lighting)
            data["data_url"] = "https://s3.amazonaws.com/%s/%s" % (
                self.bucket_name, key_data)
            data["superpixels_url"] = "https://s3.amazonaws.com/%s/%s" % (
                self.bucket_name, key_superpixels)
        else:
            mask_path = os.path.join(
                data["results_local_dir"], self.bucket_name, key_semantic)
            lighting_path = os.path.join(
                data["results_local_dir"], self.bucket_name, key_lighting)
            data_path = os.path.join(
                data["results_local_dir"], self.bucket_name, key_data)

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            os.makedirs(os.path.dirname(lighting_path), exist_ok=True)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
            superpixels_path = os.path.join(
                data["results_local_dir"], self.bucket_name, key_superpixels)

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            os.makedirs(os.path.dirname(lighting_path), exist_ok=True)
            os.makedirs(os.path.dirname(superpixels_path), exist_ok=True)

            # Convert dtypes if necessary
            if mask_image.dtype == np.float32:
                mask_image = (255 * mask_image).astype(np.uint8)
            if lighting_image.dtype == np.float32:
                lighting_image = (255 * lighting_image).astype(np.uint8)

            imsave(mask_path, mask_image)
            imsave(lighting_path, lighting_image)
            
            with open(data_path, 'w') as outfile:
                json.dump(json_dict, outfile, indent=5)

            imsave(superpixels_path, superpixels_image)

            data["semantic_url"] = "http://127.0.0.1:8080/getimage/%s/%s" % (
                self.bucket_name, key_semantic)
            data["lighting_url"] = "http://127.0.0.1:8080/getimage/%s/%s" % (
                self.bucket_name, key_lighting)
            data["data_url"] = "http://127.0.0.1:8080/getimage/%s/%s" % (
                self.bucket_name, key_data)
            print("Data url is: " + data["data_url"])
            data["superpixels_url"] = "http://127.0.0.1:8080/getimage/%s/%s" % (
                self.bucket_name, key_superpixels)
