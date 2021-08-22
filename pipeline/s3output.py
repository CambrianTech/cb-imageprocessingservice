from io import BytesIO
try:
    from imageio import imsave
except:
    from scipy.misc import imsave
import boto3
import numpy as np
import os
import json
import zlib

from pipeline.output import PipelineOutput

class PipelineS3Output(PipelineOutput):
    def __init__(self, path, s3Client:S3Client):
        super().__init__(path)
        self.s3_client = s3Client

    @property
    def required_keys(self) -> list:
        return ["semantic", "lighting", "superpixels"]

    @property
    def output_keys(self) -> list:
        return ["semantic_url", "lighting_url", "data_url" "superpixels_url"]

    def save_data(self, data, data_v3_dict):
        
        # Upload to S3 or write to local folder if local dir is set.
        if "results_local_dir" not in data:
            self.s3_client.upload_image_to_s3(mask_image,
                                self.bucket_name, key_semantic)
            self.s3_client..upload_image_to_s3(lighting_image,
                                self.bucket_name, key_lighting)
            # self.s3_client.upload_json_to_s3(json_dict,
            #                    self.bucket_name, key_data)
            # self.s3_client.upload_json_to_s3(data_v2_dict,
            #                    self.bucket_name, key_data_v2)
            self.s3_client..upload_json_to_s3(data_v3_dict,
                               self.bucket_name, key_data_v3)
            self.s3_client..upload_image_to_s3(superpixels_image,
                                self.bucket_name, key_superpixels)

            if "planes_alpha_mask" in data:
                self.s3_client.upload_image_to_s3(data["planes_alpha_mask"], self.bucket_name, key_planes_alpha_mask)

            if "planes_index_mask" in data:
                compressed_index_mask = get_compressed_index_mask(data["planes_index_mask"])
                self.s3_client.upload_bytes_to_s3(compressed_index_mask, self.bucket_name, key_planes_index_mask)

            if "planes" in data:
                for i, plane_mask in enumerate(data["planes"]["masks"]):
                    self.s3_client.upload_image_to_s3(plane_mask, self.bucket_name, "%s/plane_masks/mask_%d.png" % (unique_id, i))
        