import abc
import cv2
import numpy as np
from scipy import ndimage
from scipy.stats import mode
from enum import Enum, IntEnum

from pipeline.core import PipelineStep
from pipeline.semantics import Groupings, isolate_masks, combine_floor_masks
from pipeline.utils import resize_array
from pipeline.logging import log_image, log_segmentation_image, log_ply, im_logging_enabled, LogLevel

class Dimension(IntEnum):
    Horizontal = 0
    Vertical = 1

class PlanarDimension(metaclass=abc.ABCMeta):
    def __init__(self, indices, angles):
        self.indices = indices
        self.angles = angles

class HorizontalDimension(PlanarDimension):
    def __init__(self, floor_indices, ceiling_indices, horiz_indices, angles):
        super().__init__(horiz_indices, angles)
        self.floor_indices = floor_indices
        self.ceiling_indices = ceiling_indices

class VerticalDimension(PlanarDimension):
    def __init__(self, wall_indices, vert_indices, angles):
        super().__init__(vert_indices, angles)
        self.wall_indices = wall_indices
        

class PipelinePlaneGeometry(PipelineStep):
    def __init__(self, data=None, isolated_masks=None, image=None):
        super().__init__()

        self.data = data
        self.isolated_masks = isolated_masks
        self.image = image

    @property
    def required_keys(self) -> list:
        return ["image", "semantic_probs", "normals"]

    @property
    def output_keys(self) -> list:
        return ["output", "isolated", "floor_normal", "floor_offset", "floor_index"]

    def run(self, data):
        self.data = data
        self.image = self.data["downscaled"] if "downscaled" in self.data else self.data["image"]

        #Consolidate types: Include other types as part of floor: rug, earth, grass
        self.output = np.float32(self.data["semantic_probs"])
        combine_floor_masks(self.output)
        self.isolated_masks = isolate_masks(self.data, self.output) #break masks into major groups: Floor, Wall, Ceiling, etc

        self.process()

        data["output"] = self.output
        data["isolated"] = self.isolated_masks
        data["floor_normal"] = self.floor_normal
        data["floor_offset"] = self.floor_offset
        data["floor_index"] = self.floor_index
        
    def process(self):
        self.digest_data()
        self.calculate_geometry()
        self.cluster()

    def digest_data(self):

        shape = (self.image.shape[1], self.image.shape[0])

        planes_data = self.data["planes"]
        self.plane_parameters = np.array(planes_data["detection"][:, 6:9], dtype=np.float32)
        self.plane_offsets = np.linalg.norm(self.plane_parameters, axis=-1, keepdims=True)
        self.plane_normals = self.plane_parameters / np.maximum(self.plane_offsets, 1e-4)
        self.plane_clusters = np.array(planes_data["detection"][:, 4], dtype=np.int32)
        self.cluster_prob = planes_data["detection"][:, 5]
        self.plane_masks = resize_array(planes_data["masks"], shape)

        XYZ = planes_data["XYZ"][:, 80:-80, :].transpose(1, 2, 0)
        self.XYZ = cv2.resize(XYZ, shape)

        normals = cv2.resize(self.data["normals"], shape)

        if im_logging_enabled(self.data, LogLevel.Images):
            log_image(self.data, "xyz", 255. * self.XYZ / np.amax(self.XYZ))
            log_image(self.data, "normals", normals)

        if im_logging_enabled(self.data, LogLevel.Models):
            plane_XYZ = planes_data["plane_XYZ"][:, :, 80:-80, :].transpose(0, 2, 3, 1)
            plane_XYZ = resize_array(plane_XYZ, shape)
            log_ply(self.data, "3D", self.image, self.plane_masks, np.float32(plane_XYZ), mult=1)

        self.normals = (normals - 127.5) / 127.5
        

    def calculate_geometry(self):
        
        self.dimensions = []
        
        self.dimensions.append(self.find_floor_indices())
        self.dimensions.append(self.find_wall_indices())

        wall_indices = self.dimensions[Dimension.Vertical].wall_indices
        floor_indices = self.dimensions[Dimension.Horizontal].floor_indices
        ceiling_indices = self.dimensions[Dimension.Horizontal].ceiling_indices
        
        self.basis_indices = np.int32(np.concatenate([floor_indices, ceiling_indices, wall_indices]))

    def find_floor_indices(self):
        floor_mask = cv2.resize(self.isolated_masks[Groupings.Floor], (self.plane_masks[0].shape[1], self.plane_masks[0].shape[0]))
        ceiling_mask = cv2.resize(self.isolated_masks[Groupings.Ceiling], (self.plane_masks[0].shape[1], self.plane_masks[0].shape[0]))

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
            dots.append(np.dot(self.plane_normals[floor_indices[0]], self.plane_normals[d]))

        angs = np.arccos(np.clip(dots, -1.0, 1.0)) * 180 / np.pi

        horiz_indices = np.nonzero(np.abs(angs) < 15)[0]

        #maybe move to HorizontalDimension class:
        self.floor_normal = [0., 0, -1]
        self.floor_index = -1

        if len(floor_indices) > 0:
            self.floor_index = floor_indices[0]
            self.floor_normal = self.plane_normals[self.floor_index]
            self.floor_offset = self.plane_offsets[self.floor_index]

        ceiling_index = -1
        if len(ceiling_indices) > 0:
            self.ceiling_index = ceiling_indices[0]
            self.ceiling_normal = self.plane_normals[self.ceiling_index]
            self.floor_offset = self.plane_offsets[self.floor_index]

        return HorizontalDimension(floor_indices, ceiling_indices, horiz_indices, angs)


    def find_wall_indices(self):

        wall_mask = cv2.resize(self.isolated_masks[Groupings.Wall], (self.plane_masks[0].shape[1], self.plane_masks[0].shape[0]))

        wall_intersections = []
        dots = []
        for d in range(len(self.plane_masks)):
            wall_intersections.append(cv2.countNonZero(np.uint8(self.plane_masks[d][wall_mask > 1. / 3.] > .5)))
            dots.append(np.dot(self.floor_normal, self.plane_normals[d]))

        angs = np.arccos(np.clip(dots, -1.0, 1.0)) * 180 / np.pi
        vert_indices = np.nonzero(np.abs(90 - angs) < 15)[0]

        scores = np.int32(wall_intersections)
        wall_indices = np.nonzero(np.logical_and(scores > np.mean(scores), np.abs(90 - angs) < 15))[0]

        return VerticalDimension(wall_indices, vert_indices, angs)


    def cluster(self):
        normals_combined, normals_nn_normals = self.combined_normals()
        self.normals_c = normals_combined

        lengths = np.maximum(np.sqrt(np.sum(self.normals_c * self.normals_c, -1)), 1e-6)
        self.normals_c /= np.dstack((lengths, lengths, lengths))

        cluster_masks = [.03 * np.ones_like(self.plane_masks[0])]
        cluster_mask_indices = [0]
        vert_indices = self.dimensions[Dimension.Vertical].indices
        for i in range(1, 8):
            clust = np.nonzero(self.plane_clusters == i)[0]
            clust = np.intersect1d(clust, vert_indices)

            if len(clust) > 0:
                cluster_masks.append(np.sum(self.plane_masks[clust], 0))
                cluster_mask_indices.append((i - 1) % 3 + 1)

        if im_logging_enabled(self.data, LogLevel.Images):
            log_image(self.data, "normals_c_org", 127.5 * (self.normals_c + 1))
            plane_cluster_seg = np.argmax(cluster_masks, 0)
            cluster_mask_indices = np.int32(cluster_mask_indices)
            plane_cluster_seg_rs = np.int32(
                cv2.resize(np.uint8(plane_cluster_seg), (self.data["image"].shape[1], self.data["image"].shape[0]), interpolation=cv2.INTER_NEAREST))

            log_segmentation_image(self.data, "plane_cluster_seg", plane_cluster_seg, self.data["image"])


    def combined_normals(self):

        normals = -1 * self.normals

        plane_normals_nn = self.plane_normals.copy()
        number_planes = len(self.plane_normals)
        cluster_indices = self.cluster_prob > .5
        # print("good clusters", cluster_indices)
        basis_indices = self.basis_indices[cluster_indices[self.basis_indices]]

        for i in range(number_planes):
            if cluster_indices[i]:
                plane_normals_nn[i] = mode(normals[self.plane_masks[i] > np.amax(self.plane_masks[i]) / 2.], axis=0)[0]
                plane_normals_nn[i] /= np.linalg.norm(plane_normals_nn[i])


        # plane_normals_nn[0] = -plane_normals_nn[0]

        R, _ = self.calcTransformation(plane_normals_nn[basis_indices], self.plane_normals[basis_indices])
        # R = np.eye(3)

        nr = normals.reshape((-1, 3))
        normals = np.matmul(R, nr.transpose()).transpose().reshape(normals.shape)

        lengths = np.maximum(np.sqrt(np.sum(normals * normals, -1)), 1e-6)
        normals /= np.dstack((lengths, lengths, lengths))

        plane_normals_nn[basis_indices] = np.matmul(R,plane_normals_nn[basis_indices].transpose()).transpose()


        for i in range(number_planes):
            # if cluster_indices[i]:
            m = self.plane_masks[i].copy()
            # m[m>.5] = 1
            mult = np.dstack((m,m,m))

            normals = (1.0 - mult) * normals + mult * self.plane_normals[i]

        lengths = np.maximum(np.sqrt(np.sum(normals * normals, -1)), 1e-6)
        normals /= np.dstack((lengths, lengths, lengths))
        return normals, plane_normals_nn
        
    def calcTransformation(self, points_1, points_2):
        # center_1 = points_1.mean(0) =(0,0,0)
        # center_2 = points_2.mean(0)=(0,0,0)
        center_1 = (0, 0, 0)
        center_2 = (0, 0, 0)
        H = np.matmul((points_1 - center_1).transpose(), (points_2 - center_2))
        U, S, V = np.linalg.svd(H)

        R = np.matmul(V.transpose(), U.transpose())
        if np.linalg.det(R) < 0 and False:
            R[:, 2] *= -1
            pass
        t = -np.matmul(R, center_1) + center_2
        return R, t



        
        


