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

surface_types = ["unknown", "floor", "wall", "horizontal", "vertical"]

@abstract
class PipelineOutput(PipelineStep):
    def __init__(self, bucket_name):
        super().__init__()
        self.bucket_name = bucket_name
        self.s3_client = boto3.client("s3")

    @property
    def output_keys(self) -> list:
        return ["semantic_url", "lighting_url", "data_url" "superpixels_url"]

    def run(self, data):
        unique_id = data["unique_id"]
        key_semantic = "%s/mask.png" % unique_id
        key_lighting = "%s/lighting.png" % unique_id
        key_data = "%s/data.json" % unique_id
        key_data_v2 = "%s/data_v2.json" % unique_id
        key_data_v3 = "%s/data_v3.json" % unique_id
        key_superpixels = "%s/superpixels.png" % unique_id
        key_planes_index_mask = "%s/planes_index_mask.zz" % unique_id
        key_planes_alpha_mask = "%s/planes_alpha_mask.png" % unique_id

        # Use AWS S3 url by default, or local server if one was set.
        base_url = "http://127.0.0.1:8080/getimage" if "results_local_dir" in data else "https://s3.amazonaws.com"

        def _make_url(path):
            return "%s/%s/%s" % (base_url, self.bucket_name, path)

        def _make_plane_mask_url(plane_index):
            return _make_url("%s/plane_masks/mask_%d.png" % (unique_id, plane_index))

        mask_image = data["mask"]
        lighting_image = data["lighting"]
        superpixels_image = data["superpixels"]

        # json_dict = _make_data_dict(data, _make_url)
        # data_v2_dict = _make_data_v2_dict(data,
        #                                   _make_url(key_lighting),
        #                                   _make_url(key_superpixels),
        #                                   _make_url(key_semantic),
        #                                   _make_url(key_planes_index_mask),
        #                                   _make_url(key_planes_alpha_mask),
        #                                   _make_plane_mask_url)

        data_v3_dict = self.make_data_v3_dict(data,
                                          _make_url(key_lighting),
                                          _make_url(key_superpixels),
                                          _make_url(key_semantic),
                                          _make_url(key_planes_index_mask),
                                          _make_url(key_planes_alpha_mask),
                                          _make_plane_mask_url)

        # Upload to S3 or write to local folder if local dir is set.
        self.save_data(data_v3_dict)

        data["semantic_url"] = _make_url(key_semantic)
        data["lighting_url"] = _make_url(key_lighting)
        # data["data_url"] = _make_url(key_data)
        # data["data_v2_url"] = _make_url(key_data_v2)
        data["data_v3_url"] = _make_url(key_data_v3)
        data["superpixels_url"] = _make_url(key_superpixels)

    @abstractmethod
    def save_data(self, data, data_v3_dict):
        print("")
    

    def make_data_dict(self, data, make_url):
        return {
            "formatVersion": 1,
            "cameraPosition": [0.0, data["camera_elevation"], 0.0],
            "cameraRotation": data["camera_rotation"],
            "floorRotation": data["floor_rotation"],
            "fov": data["fov"]
        }


    def encode_plane_surface(self, plane_index, plane_data, mask_url):
        # https://github.com/NVlabs/planercnn#plane-representation
        # In this project, plane parameters are of absolute scale (in terms of meters).
        # Each plane has three parameters, which equal to plane_normal * plane_offset.
        # Suppose plane_normal is (a, b, c) and plane_offset is d, every point (X, Y, Z)
        # on the plane satisfies, aX + bY + cZ = d.
        # Then plane parameters are (a, b, c)*d. Since plane normal is a unit vector,
        # we can extract plane_normal and plane_offset from their multiplication.
        plane_parameters = np.array(plane_data[6:9], dtype=np.float32)
        plane_offset = np.maximum(1e-4, np.linalg.norm(plane_parameters))
        plane_normal = plane_parameters / plane_offset

        plane_image_extents = {
            "minY": plane_data[0],
            "minX": plane_data[1],
            "maxY": plane_data[2],
            "maxX": plane_data[3]
        }

        plane_type = np.uint8(plane_data[9])


        return {
            "id": "plane-%d" % plane_index,
            "type": surface_types[plane_type],
            "name": "Plane %d" % plane_index,
            "normal": plane_normal.tolist(),
            "offset": plane_offset,
            "rawParams": plane_parameters.tolist(),
            "imageExtents": plane_image_extents,
            "images": {
                "mask": mask_url
            }
        }


    def make_data_v2_dict(self, data, lighting_url, superpixels_url, semantic_url, planes_index_mask_url, planes_alpha_mask_url, make_plane_mask_url):
        all_plane_data = data["planes"]["detection"].tolist(
        ) if "planes" in data else []

        return {
            "formatVersion": 2,
            "name": "Room %s" % data["unique_id"],
            "id": "room-%s" % data["unique_id"],
            "floorRotation": data["floor_rotation"],
            "images": {
                "lighting": lighting_url,
                "superpixels": superpixels_url,
                "semantic": semantic_url,
                "planes_alpha_mask": planes_alpha_mask_url
            },
            "compressed": {
                "planes_index_mask": planes_index_mask_url,
            },
            "camera": {
                "fov": data["fov"],
                "position": [0, 0, 0],  # Planes are relative to camera so the
                "rotation": [0, 0, 0]  # camera is all zeros.
            },
            "geometry": {
                "surfaces": [
                    self.encode_plane_surface(i, plane_data, make_plane_mask_url(i))
                    for i, plane_data in enumerate(all_plane_data)
                ]
            },
            "assets": []
        }


    def encode_plane_surface_v3(self, plane_index, plane_data, contour_data, mask_url):
        # https://github.com/NVlabs/planercnn#plane-representation
        # In this project, plane parameters are of absolute scale (in terms of meters).
        # Each plane has three parameters, which equal to plane_normal * plane_offset.
        # Suppose plane_normal is (a, b, c) and plane_offset is d, every point (X, Y, Z)
        # on the plane satisfies, aX + bY + cZ = d.
        # Then plane parameters are (a, b, c)*d. Since plane normal is a unit vector,
        # we can extract plane_normal and plane_offset from their multiplication.
        plane_parameters = np.array(plane_data[6:9], dtype=np.float32)
        plane_offset = np.maximum(1e-4, np.linalg.norm(plane_parameters))
        plane_normal = plane_parameters / plane_offset
        plane_rotation = plane_data[10]

        plane_image_extents = {
            "minY": plane_data[0],
            "minX": plane_data[1],
            "maxY": plane_data[2],
            "maxX": plane_data[3]
        }

        plane_type = np.uint8(plane_data[9])


        return {
            "id": "plane-%d" % plane_index,
            "type": surface_types[plane_type],
            "name": "Plane %d" % plane_index,
            "normal": plane_normal.tolist(),
            "offset": plane_offset,
            "rotation": plane_rotation,
            "rawParams": plane_parameters.tolist(),
            "imageExtents": plane_image_extents,
            "images": {
                "mask": mask_url
            },
            "contours": contour_data
        }


    def make_data_v3_dict(self, data, lighting_url, superpixels_url, semantic_url, planes_index_mask_url, planes_alpha_mask_url, make_plane_mask_url):
        all_plane_data = data["planes"]["detection"].tolist(
        ) if "planes" in data else []

        contour_plane_data = data["planes"]["contours"] if "planes" in data else []

        return {
            "formatVersion": 3,
            "name": "Room %s" % data["unique_id"],
            "id": "room-%s" % data["unique_id"],
            "floorRotation": data["floor_rotation"],
            "images": {
                "lighting": lighting_url,
                "superpixels": superpixels_url,
                "semantic": semantic_url,
                "planes_alpha_mask": planes_alpha_mask_url
            },
            "compressed": {
                "planes_index_mask": planes_index_mask_url,
            },
            "camera": {
                "fov": data["fov"],
                "position": [0, 0, 0],  # Planes are relative to camera so the
                "rotation": [0, 0, 0]  # camera is all zeros.
            },
            "geometry": {
                "surfaces": [
                    self.encode_plane_surface_v3(i, plane_data, contour_data, self.make_plane_mask_url(i)) for i, (plane_data, contour_data
                                                                                    ) in enumerate(zip(all_plane_data, contour_plane_data))
                ]
            },
            "assets": []
        }


    def get_compressed_index_mask(self, index_mask):
        # Convert int16 to two uint8
        assert index_mask.dtype == np.int16
        index_mask_bytes = index_mask.tobytes()

        # Compress the bytes with zlib (deflate).
        compress = zlib.compressobj()
        compressed_index_mask_bytes = compress.compress(
            index_mask_bytes
        )
        compressed_index_mask_bytes += compress.flush()

        return compressed_index_mask_bytes