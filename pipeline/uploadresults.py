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
        super().__init__()
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")

    @property
    def required_keys(self) -> list:
        return ["semantic", "lighting", "superpixels", "planes"]

    @property
    def output_keys(self) -> list:
        return ["semantic_url", "lighting_url", "data_url" "superpixels_url"]

    def run(self, data):
        key_semantic = "%s/mask.png" % data["image_s3_key"]
        key_lighting = "%s/lighting.png" % data["image_s3_key"]
        key_data = "%s/data.json" % data["image_s3_key"]
        key_superpixels = "%s/superpixels.png" % data["image_s3_key"]
        
        # Use AWS S3 url by default, or local server if one was set.
        base_url = "http://127.0.0.1:8080/getimage" if "results_local_dir" in data else "https://s3.amazonaws.com"
        def _make_url(path):
            return "%s/%s/%s" % (base_url, self.bucket_name, path)

        json_dict = {
            "cameraPosition": [0.0, data["camera_elevation"], 0.0],
            "cameraRotation": data["camera_rotation"],
            "floorRotation": data["floor_rotation"],
            "fov": data["fov"],
            "planes": [{
                "data": plane_data, # [9]
                "mask_url": _make_url("%s/plane_masks/mask_%d.png" % (data["image_s3_key"], i))
            } for i, plane_data in enumerate(data["planes"]["detections"].tolist())]
        }

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

            for i, plane_mask in enumerate(data["planes"]["masks"]):
                _upload_image_to_s3(self.s3_client, plane_mask, self.bucket_name, "%s/plane_masks/mask_%d.png" % (data["image_s3_key"], i))
        else:
            def _make_local_url(path):
                return os.path.join(data["results_local_dir"], self.bucket_name, path)

            mask_path = _make_local_url(key_semantic)
            lighting_path = _make_local_url(key_lighting)
            data_path = _make_local_url(key_data)
            superpixels_path = _make_local_url(key_superpixels)

            os.makedirs(os.path.dirname(mask_path), exist_ok=True)
            os.makedirs(os.path.dirname(lighting_path), exist_ok=True)
            os.makedirs(os.path.dirname(data_path), exist_ok=True)
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

            for i, plane_mask in enumerate(data["planes"]["masks"]):
                plane_path = _make_local_url("%s/plane_masks/mask_%d.png" % (data["image_s3_key"], i))
                os.makedirs(os.path.dirname(plane_path), exist_ok=True)
                if plane_mask.dtype == np.float32:
                    plane_mask = (255 * plane_mask).astype(np.uint8)
                imsave(plane_path, plane_mask)

        data["semantic_url"] = _make_url(key_semantic)
        data["lighting_url"] = _make_url(key_lighting)
        data["data_url"] = _make_url(key_data)
        data["superpixels_url"] = _make_url(key_superpixels)