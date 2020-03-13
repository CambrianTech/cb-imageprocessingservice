from pipeline.core import PipelineStep
import cv2
import numpy as np
import math
from cambrian import image_processing as ip, transformations as T, geometry as geo



class PipelineDeterminePrimaryAngles(PipelineStep):
    @property
    def required_keys(self) -> list:
        return []

    @property
    def output_keys(self) -> list:
        return ["camera_rotation", "camera_elevation", "floor_rotation"]

    def run(self, data):

        cam_pitch = -0.2
        cam_yaw = 0.0
        cam_roll = 0.0

        data["camera_rotation"] = [cam_pitch, cam_yaw, cam_roll]

        print("camera rotation:", data["camera_rotation"])

        data["camera_elevation"] = 1.3

        print("floor elevation:", data["camera_elevation"])

        data["floor_rotation"] = 0.0

        print("floor rotation:", data["floor_rotation"])
