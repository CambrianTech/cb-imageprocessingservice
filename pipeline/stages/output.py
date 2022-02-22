from abc import abstractmethod
from io import BytesIO

import boto3
import numpy as np
import os
import json
import zlib
import cv2

from pipeline.core import PipelineStep, PipelineStepIndex

surface_types = ["unknown", "floor", "wall", "horizontal", "vertical"]

class PipelineOutput(PipelineStep):
    def __init__(self, pipeline, outfile_name="data.json", preview_size=1024, thumbnail_size=320):
        super().__init__(pipeline)
        self.outfile_name = outfile_name
        self.preview_size = preview_size
        self.thumbnail_size = thumbnail_size
        self.y_up = self.config.api_level > 3

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Output

    @property
    def required_keys(self) -> list:
        if self.config.api_level == 1:
            return ["mask", "lighting", "superpixels"]
        elif self.config.api_level == 2:
            return ["lighting", "superpixels"]
        elif self.config.api_level == 3:
            return ["planes", "lighting"]

        return ["planes", "lighting", "planes_alpha_mask", "planes_index_mask"]

    @property
    def output_keys(self) -> list:

        if self.config.api_level == 1:
            return ["version", "data_url", "lighting_url", "superpixels_url", "mask"]
        elif self.config.api_level < 3:
            return ["version", "data_url", "lighting_url", "superpixels_url"]

        return ["version", "data_url"]

    def make_url(self, filename, directory=None):
        if directory is None:
            return filename
        return os.path.join(directory, filename)

    def make_plane_mask_url(self, plane_index):
        return self.make_url("mask_%d.png" % plane_index, "plane_masks")

    def run(self, data):
        self.unique_id = data["unique_id"]

        data["version"] = self.config.api_level
        data["data_url"] = self.make_url(self.outfile_name)

        filename = "background.jpg"
        image_url = self.make_url(filename)
        self.save_image(data["image"], filename, image_url)

        def scale_to_constraint(image, size, interpolation=cv2.INTER_AREA):
            scale = min(size / image.shape[0], size / image.shape[1])
            return cv2.resize(image, (int(scale * image.shape[1]), int(scale * image.shape[0])), interpolation)

        filename = "preview.jpg"
        preview_url = self.make_url(filename)
        self.save_image(scale_to_constraint(data["image"], self.preview_size), filename, preview_url, 70)

        filename = "thumbnail.jpg"
        thumbnail_url = self.make_url(filename)
        self.save_image(scale_to_constraint(data["image"], self.thumbnail_size), filename, thumbnail_url, 60)

        if self.config.api_level == 1:
            filename = "mask.png"
            mask_url = self.make_url(filename)
            self.save_image(data["mask"], filename, mask_url)
            results = self.make_data_dict(data)
        else:
            filename = "lighting.png"
            lighting_url = self.make_url(filename)
            self.save_image(data["lighting"], filename, lighting_url)
            data["lighting_url"] = lighting_url

            if self.config.api_level == 2:
                filename = "superpixels.png"
                superpixels_url = self.make_url(filename)
                self.save_image(data["superpixels"], filename, superpixels_url)
                data["superpixels_url"] = self.make_url(superpixels_url)


            if self.config.api_level == 2 or self.config.api_level == 3:            
                if "masks" in data:
                    for i, plane_mask in enumerate(data["masks"]):
                        mask_url = self.make_plane_mask_url(i)
                        self.save_image(plane_mask, filename, mask_url)

                if self.config.api_level == 2:
                    results = self.make_data_v2_dict(data, image_url, lighting_url, superpixels_url)
                else:
                    results = self.make_data_v3_dict(data, image_url, lighting_url)
            else:

                if "planes_index_mask" in data:
                    filename = "planes_index_mask.zz"
                    planes_index_mask_url = self.make_url(filename)
                    image = self.get_compressed_index_mask(data["planes_index_mask"])
                    self.save_file(image, filename, planes_index_mask_url)

                    filename = "planes_alpha_mask.png"
                    planes_alpha_mask_url = self.make_url(filename)
                    self.save_image(data["planes_alpha_mask"], filename, planes_alpha_mask_url)


                #save masks independently temporarily
                surfaces = data["room"].surfaces
                for i in range(len(surfaces)):
                    surface = surfaces[i]
                    mask_url = self.make_plane_mask_url(i)
                    mask = surface.mask if surface.final_mask is None else surface.final_mask
                    mask = mask * 255
                    scale = 1500 / min(mask.shape[0], mask.shape[1])
                    if scale > 1:
                        mask = cv2.resize(mask, (int(scale * mask.shape[1]), int(scale * mask.shape[0])))
                    self.save_image(mask, filename, mask_url)

                results = self.make_data_v4_dict(data, image_url, lighting_url, planes_index_mask_url, planes_alpha_mask_url)

        results["data_url"] = data["data_url"]

        self.save_data(results, self.outfile_name, data["data_url"])

    @abstractmethod
    def save_image(self, image, filename, url, quality=None):
        return

    @abstractmethod
    def save_file(self, bytes, filename, url):
        return

    @abstractmethod
    def save_data(self, data, filename, url):
        return

    def make_data_dict(self, data):
        return {
            "version": self.config.api_level,
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
        plane_normal = (plane_parameters / plane_offset).tolist()

        plane_type = np.uint8(plane_data[9])

        return {
            "id": "plane-%d" % plane_index,
            "type": surface_types[plane_type],
            "name": "Plane %d" % plane_index,
            "normal": [-plane_normal[0], -plane_normal[2], plane_normal[1]] if self.y_up else plane_normal,
            "offset": plane_offset,
            "images": {"mask": mask_url}
        }


    def make_data_v2_dict(self, data, image_url, lighting_url, superpixels_url):
        all_plane_data = data["planes"]["detection"].tolist(
        ) if "planes" in data else []

        return {
            "formatVersion": 2,
            "name": "Room %s" % self.unique_id,
            "id": "room-%s" % self.unique_id,
            "floorRotation": data["floor_rotation"],
            "images": {
                "main": image_url,
                "lighting": lighting_url,
                "superpixels": superpixels_url
            },
            "camera": {
                "fov": data["fov"],
                "position": [0, 0, 0],  # Planes are relative to camera so the
                "rotation": [0, 0, 0]  # camera is all zeros.
            },
            "geometry": {
                "surfaces": [
                    self.encode_plane_surface(i, plane_data, self.make_plane_mask_url(i)) for i, plane_data in enumerate(all_plane_data)
                ]
            },
            "assets": []
        }


    def encode_plane_surface_v3(self, plane_index, plane_data, mask_url):
        # https://github.com/NVlabs/planercnn#plane-representation
        # In this project, plane parameters are of absolute scale (in terms of meters).
        # Each plane has three parameters, which equal to plane_normal * plane_offset.
        # Suppose plane_normal is (a, b, c) and plane_offset is d, every point (X, Y, Z)
        # on the plane satisfies, aX + bY + cZ = d.
        # Then plane parameters are (a, b, c)*d. Since plane normal is a unit vector,
        # we can extract plane_normal and plane_offset from their multiplication.
        plane_parameters = np.array(plane_data[6:9], dtype=np.float32)
        plane_offset = np.maximum(1e-4, np.linalg.norm(plane_parameters))
        plane_normal = (plane_parameters / plane_offset).tolist()

        plane_rotation = 0
        plane_type = 0

        if len(plane_data) > 9:
            plane_rotation = plane_data[10]
            plane_type = np.uint8(plane_data[9])


        return {
            "id": "plane-%d" % plane_index,
            "type": surface_types[plane_type],
            "name": "Plane %d" % plane_index,
            "normal": [-plane_normal[0], -plane_normal[2], plane_normal[1]] if self.y_up else plane_normal,
            "offset": plane_offset,
            "rotation": plane_rotation,
            "images": {
                "mask": mask_url
            }
        }

    def encode_plane_surface_v4(self, plane_index, surface, mask_url):
        # https://github.com/NVlabs/planercnn#plane-representation
        # In this project, plane parameters are of absolute scale (in terms of meters).
        # Each plane has three parameters, which equal to plane_normal * plane_offset.
        # Suppose plane_normal is (a, b, c) and plane_offset is d, every point (X, Y, Z)
        # on the plane satisfies, aX + bY + cZ = d.
        # Then plane parameters are (a, b, c)*d. Since plane normal is a unit vector,
        # we can extract plane_normal and plane_offset from their multiplication.

        plane_normal = surface.normal.astype(float)
        plane_normal = [-plane_normal[0], -plane_normal[2], plane_normal[1]] if self.y_up else list(plane_normal)
        plane_offset = float(surface.offset)

        return {
            "id": str(surface.uniqueId),
            "type": surface.surfaceType.name,
            "name": surface.name,
            "normal": plane_normal,
            "offset": plane_offset,
            "axisRotation": -surface.axisRotation if self.y_up else surface.axisRotation,
            "images": {
                "mask": mask_url
            }
        }


    def make_data_v3_dict(self, data, image_url, lighting_url):
        all_plane_data = data["planes"]["detection"].tolist(
        ) if "planes" in data else []

        return {
            "formatVersion": 3,
            "name": "Room %s" % self.unique_id,
            "id": "room-%s" % self.unique_id,
            "floorRotation": data["floor_rotation"],
            "images": {
                "main": image_url,
                "lighting": lighting_url,
            },
            "camera": {
                "fov": data["fov"],
                "position": [0, 0, 0],  # Planes are relative to camera so the
                "rotation": [0, 0, 0]  # camera is all zeros.
            },
            "geometry": {
                "surfaces": [
                    self.encode_plane_surface_v3(i, plane_data, self.make_plane_mask_url(i)) for i, (plane_data) in enumerate(all_plane_data)
                ]
            },
            "assets": []
        }

    def make_data_v4_dict(self, data, image_url, lighting_url, planes_index_mask_url, planes_alpha_mask_url):

        # surfaces_json = self.encode_plane_surface_v4(0, data["room"].surfaces[0], self.make_plane_mask_url(0))
        # print("surfaces_json 4:\n", surfaces_json)

        # all_plane_data = data["planes"]["detection"].tolist() if "planes" in data else []
        # surfaces_json = self.encode_plane_surface_v3(0, all_plane_data[0], self.make_plane_mask_url(0))
        # print("surfaces_json 3:\n", surfaces_json)

        return {
            "formatVersion": 4,
            "version": "4.0.1",
            "name": "Room %s" % self.unique_id,
            "id": self.unique_id,
            "floorRotation": -data["floor_rotation"] if self.y_up else data["floor_rotation"],
            "images": {
                "main": image_url,
                "lighting": lighting_url,
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
                "verticalAxis": "y" if self.y_up else "z",
                "surfaces": [
                    self.encode_plane_surface_v4(i, surface, self.make_plane_mask_url(i)) for i, (surface) in enumerate(data["room"].surfaces)
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