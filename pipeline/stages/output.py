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
        return ["planes", "lighting", "index_mask"]

    @property
    def output_keys(self) -> list:
        return ["version", "data_url"]

    def make_url(self, filename, directory=None):
        if directory is None:
            return filename
        return os.path.join(directory, filename)

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

        filename = "lighting.png"
        lighting_url = self.make_url(filename)
        self.save_image(data["lighting"], filename, lighting_url)
        data["lighting_url"] = lighting_url
        
        filename = "index_mask.png"
        index_mask_url = self.make_url(filename)
        self.save_image(data["index_mask"], filename, index_mask_url)    

        results = self.make_data_dict(data, image_url, lighting_url, index_mask_url)

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

    def encode_plane_surface(self, plane_index, surface):
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
            "maskIndex": plane_index + 1,
            "normal": plane_normal,
            "offset": plane_offset,
            "axisRotation": -surface.axisRotation if self.y_up else surface.axisRotation
        }

    def make_data_dict(self, data, image_url, lighting_url, index_mask_url):

        return {
            "version": "%d.0" % self.config.api_level,
            "name": "Room %s" % self.unique_id,
            "id": self.unique_id,
            "floorRotation": -data["floor_rotation"] if self.y_up else data["floor_rotation"],
            "images": {
                "main": image_url,
                "lighting": lighting_url,
                "index_mask": index_mask_url
            },
            "camera": {
                "fov": data["fov"],
                "position": [0, 0, 0],  # Planes are relative to camera so the
                "rotation": [0, 0, 0]  # camera is all zeros.
            },
            "geometry": {
                "verticalAxis": "y" if self.y_up else "z",
                "surfaces": [
                    self.encode_plane_surface(i, surface) for i, (surface) in enumerate(data["room"].surfaces)
                ]
            },
            "assets": []
        }