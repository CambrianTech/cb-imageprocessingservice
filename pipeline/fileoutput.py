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

from pipeline.core import PipelineStep

@abstract
class PipelineFileOutput(PipelineStep):
    def __init__(self, base_path):
        super().__init__()
        self.base_path = base_path

    @property
    def required_keys(self) -> list:
        return ["semantic", "lighting", "superpixels"]

    @property
    def output_keys(self) -> list:
        return ["semantic_url", "lighting_url", "data_url" "superpixels_url"]

    def save_data(self, data, data_v3_dict):
        
        # Upload to S3 or write to local folder if local dir is set.
        
        def _make_local_url(path):
            return os.path.join(data["results_local_dir"], self.bucket_name, path)

        mask_path = _make_local_url(key_semantic)
        lighting_path = _make_local_url(key_lighting)
        # data_path = _make_local_url(key_data)
        # data_v2_path = _make_local_url(key_data_v2)
        data_v3_path = _make_local_url(key_data_v3)
        superpixels_path = _make_local_url(key_superpixels)
        planes_index_mask_path = _make_local_url(key_planes_index_mask)
        planes_alpha_mask_path = _make_local_url(key_planes_alpha_mask)

        os.makedirs(os.path.dirname(mask_path), exist_ok=True)
        os.makedirs(os.path.dirname(lighting_path), exist_ok=True)
        # os.makedirs(os.path.dirname(data_path), exist_ok=True)
        # os.makedirs(os.path.dirname(data_v2_path), exist_ok=True)
        os.makedirs(os.path.dirname(data_v3_path), exist_ok=True)
        os.makedirs(os.path.dirname(superpixels_path), exist_ok=True)

        # Convert dtypes if necessary
        if mask_image.dtype == np.float32:
            mask_image = (255 * mask_image).astype(np.uint8)
        if lighting_image.dtype == np.float32:
            lighting_image = (255 * lighting_image).astype(np.uint8)

        imsave(mask_path, mask_image)
        imsave(lighting_path, lighting_image)

        # with open(data_path, "w", encoding="utf-8") as out_file:
        #     json.dump(json_dict, out_file, indent=4)
        # with open(data_v2_path, "w", encoding="utf-8") as out_file:
        #     json.dump(data_v2_dict, out_file, indent=4)
        with open(data_v3_path, "w", encoding="utf-8") as out_file:
            json.dump(data_v3_dict, out_file, indent=4)

        imsave(superpixels_path, superpixels_image)

        if "planes_alpha_mask" in data:
            alpha_mask = data["planes_alpha_mask"]
            if alpha_mask.dtype == np.float32:
                alpha_mask = (255 * alpha_mask).astype(np.uint8)
            imsave(planes_alpha_mask_path, alpha_mask)

        if "planes_index_mask" in data:
            compressed_index_mask = get_compressed_index_mask(
                data["planes_index_mask"])
            with open(planes_index_mask_path, "wb") as out_file:
                out_file.write(compressed_index_mask)

        if "planes" in data:
            for i, plane_mask in enumerate(data["planes"]["masks"]):
                plane_path = _make_local_url(
                    "%s/plane_masks/mask_%d.png" % (unique_id, i))
                os.makedirs(os.path.dirname(plane_path), exist_ok=True)
                if plane_mask.dtype == np.float32:
                    plane_mask = (255 * plane_mask).astype(np.uint8)
                imsave(plane_path, plane_mask)
