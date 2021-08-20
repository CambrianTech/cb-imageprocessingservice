import cv2
import numpy as np
from scipy import ndimage
from enum import Enum, IntEnum

class SemanticKey(Enum):
    Wall = "wall"
    Floor = "floor"
    Ceiling = "ceiling"
    WallLike = "wall-like"
    Other = "other"

class Dimension(IntEnum):
    Horizontal = 0
    Vertical = 1

#@abstract
class PlanarDimension:
    def __init__(self, indices, angles):
        self.indices = indices
        self.angles = angles

class HorizontalDimension(PlanarDimension):
    def __init__(self, floor_indices, ceiling_indices, horiz_indices, angles):
        super().__init__(horiz_indices, angles)
        self.floor_indices = floor_indices
        self.ceiling_indices = ceiling_indices

class VerticalDimension(PlanarDimension):
    def __init__(self, wall_indices, vert_indices, angs):
        super().__init__(vert_indices, angles)
        self.wall_indices = ceiling_indices
        

class PlaneGeometry:
    def __init__(self, data, isolated_masks, shape):
        super().__init__()

        self.isolated_masks = isolated_masks

        #self.digest_data(data)
        #self.calculate_geometry()


    def digest_data(self, data, shape):

        planes_data = data["planes"]
        self.plane_parameters = np.array(planes_data["planes"]["detection"][:, 6:9], dtype=np.float32)
        self.plane_offsets = np.linalg.norm(self.plane_parameters, axis=-1, keepdims=True)
        self.plane_normals = self.plane_parameters / np.maximum(self.plane_offsets, 1e-4)
        self.plane_clusters = np.array(planes_data["planes"]["detection"][:, 4], dtype=np.int32)
        # roi = np.array(data["planes"]["detection"][:, 0:4], dtype=np.int32)
        self.cluster_prob = data["planes"]["detection"][:, 5]

        self.plane_masks = planes_data["masks"]
        self.number_planes = len(self.plane_masks)
        self.plane_rotations = np.zeros(self.number_planes, dtype=np.float32)
        self.floor_rotation = 0.0

        self.plane_masks = resize_array(self.plane_masks, shape)

        plane_XYZ = pself.lanes_data["plane_XYZ"][:, :, 80:-80, :].transpose(0, 2, 3, 1)
        self.plane_XYZ = resize_array(plane_XYZ, shape)

        # depth = planes_data["depth_np"][:, 80:-80, :].transpose(1, 2, 0)
        XYZ = planes_data["XYZ"][:, 80:-80, :].transpose(1, 2, 0)
        # depth = cv2.resize(depth, shape)
        self.XYZ = cv2.resize(XYZ, shape)            

        normals = cv2.resize(data["normals"], shape)

        if im_logging_enabled(data, LogLevel.Images):
            log_image(data, "XYZ", 255. * XYZ / np.amax(XYZ))
            log_image(data, "normals", normals)

        #normalize around zero forward: -127 to 127
        self.normals = (normals - 127.5) / 127.5

    def calculate_geometry(self):
        
        self.plane_normals = plane_normals

        self.dimensions = []
        self.dimensions.append(find_floor_indices())
        self.dimensions.append(find_wall_indices())

        self.floor_normal = [0., 0, -1]
        self.floor_index = -1

        floor_indices = self.dimensions[Dimension.Horizontal].floor_indices

        if len(floor_indices) > 0:
            self.floor_index = floor_indices[0]

            self.floor_normal = plane_normals[floor_index]
            self.floor_offset = plane_offsets[floor_index]
            # print("floor_normal", floor_normal)

    def find_floor_indices(self):
        # floor_mask = cv2.resize(floor_mask, (plane_masks[0].shape[1], plane_masks[0].shape[0]))

        floor_mask = cv2.resize(self.isolated_masks[SemanticKey.Floor], (self.plane_masks[0].shape[1], self.plane_masks[0].shape[0]))
        ceiling_mask = cv2.resize(self.isolated_masks[SemanticKey.Ceiling], (self.plane_masks[0].shape[1], self.plane_masks[0].shape[0]))

        floor_intersections = []
        ceiling_intersections = []

        dots = []
        for d in range(len(self.plane_masks)):
            floor_intersections.append(cv2.countNonZero(np.uint8(self.plane_masks[d][floor_mask > np.amax(floor_mask) / 2.] > np.amax(self.plane_masks[d]) / 2.)))
            ceiling_intersections.append(cv2.countNonZero(np.uint8(self.plane_masks[d][ceiling_mask > np.amax(ceiling_mask) / 2.] > np.amax(self.plane_masks[d]) / 2.)))

        scores = np.int32(floor_intersections)
        floor_indices = np.nonzero(scores > np.mean(scores))[0]
        floor_indices = floor_indices[np.argsort(scores[floor_indices])[::-1]]

        scores = np.int32(ceiling_intersections)
        ceiling_indices = np.nonzero(scores > np.mean(scores))[0]
        ceiling_indices = ceiling_indices[np.argsort(scores[ceiling_indices])[::-1]]

        for d in range(len(self.plane_masks)):
            dots.append(np.dot(plane_normals[floor_indices[0]], plane_normals[d]))

        angs = np.arccos(np.clip(dots, -1.0, 1.0)) * 180 / np.pi

        horiz_indices = np.nonzero(np.abs(angs) < 15)[0]

        return HorizontalDimension(floor_indices, ceiling_indices, horiz_indices, angs)


    def find_wall_indices(self):

        wall_mask = cv2.resize(self.isolated_masks[SemanticKey.Wall], (self.plane_masks[0].shape[1], self.plane_masks[0].shape[0]))

        wall_intersections = []
        dots = []
        for d in range(len(self.plane_masks)):
            wall_intersections.append(cv2.countNonZero(np.uint8(self.plane_masks[d][wall_mask > 1. / 3.] > .5)))
            dots.append(np.dot(floor_normal, plane_normals[d]))

        angs = np.arccos(np.clip(dots, -1.0, 1.0)) * 180 / np.pi
        vert_indices = np.nonzero(np.abs(90 - angs) < 15)[0]

        scores = np.int32(wall_intersections)
        wall_indices = np.nonzero(np.logical_and(scores > np.mean(scores), np.abs(90 - angs) < 15))[0]

        return VerticalDimension(wall_indices, vert_indices, angs)

