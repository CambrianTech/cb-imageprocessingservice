from io import BytesIO
import boto3
import numpy as np
import os
try:
    from imageio import imread, imsave
except:
    from scipy.misc import imread, imsave

class S3Client:
    def __init__(self, base_url="https://s3.amazonaws.com"):
        super().__init__()
        self.base_url = base_url
        self.s3_client = boto3.client("s3")

    def get_image_from_s3(self, bucket: str, key: str) -> np.ndarray:
        data = BytesIO()
        self.s3_client.download_fileobj(bucket, "%s/background" % key, data)
        data.seek(0)
        return imread(data)  # uint8 [0, 255]

    def upload_image_to_s3(self, image: np.ndarray, bucket: str, key: str):
        if image.dtype == np.float32:
            image = (255 * image).astype(np.uint8)

        image_data = BytesIO()
        imsave(image_data, image, format=".png")
        image_data.seek(0)
        self.s3_client.upload_fileobj(image_data, bucket, key)

    def upload_bytes_to_s3(self, data: bytes, bucket: str, key: str):
        data_io = BytesIO()
        data_io.write(data)
        data_io.seek(0)
        self.s3_client.upload_fileobj(data_io, bucket, key)

    def upload_json_to_s3(self, json_dict: dict, bucket: str, key: str):
        json_data = json.dumps(json_dict, indent=4).encode("utf-8")
        self.upload_bytes_to_s3(json_data, bucket, key)


    def upload_text_to_s3(self, text: str, bucket: str, key: str):
        text_data = text.encode("utf-8")
        self.upload_bytes_to_s3(text_data, bucket, key)
