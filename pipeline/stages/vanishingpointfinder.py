import numpy as np
from scipy import ndimage
import cv2
import math
import time
from enum import IntEnum
from scipy.spatial import distance

from cambrian.LineFunctions import LineFunctions
from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.surface_type import SurfaceType
from pipeline.misc.utils import resize_array, random_color, overlay_mask, partition
from pipeline.components.base_process import BaseProcess, Multiprocessor
from .planegeometry import Dimension
from .extractsurfaces import box_like, legged_objects
from pipeline.data.logging import log_image, log_segmentation_image, im_logging_enabled
from pipeline.components.line import Line, line_on_image_edge, merge_lines, draw_lines
from pipeline.data.ade20k import ADE20K

pi_2 = np.pi/2

def angle_with_vp(model, locations, directions):

    vp = model[:2] / model[2]

    est_directions = locations - vp

    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
               np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = np.abs(dot_prod / abs_prod)

    return cosine_theta

class VanishingPoint:
    def __init__(self, lines, model, votes):

        self.lines = lines
        self.model = model
        self.votes = votes

        self._score = None
        self._inliers = None
        self._inlier_lines = None

        self._points = None
        #cv2.minAreaRect(InputArray  points)

    def __lt__(self, other):
        return self.score < other.score

    @property
    def score(self):
        if self._score is None:
            self._score = sum(self.votes)

            # if self.measure_area and len(self.points) > 1:
            #     rect = cv2.minAreaRect(self.points)
            #     self._score = self._score * np.hypot(rect[1][0], rect[1][1])

        return self._score

    @property
    def inliers(self):
        if self._inliers is None:
            self._inliers = np.array(self.lines)[self.votes > 0]

        return self._inliers

    @property
    def points(self):
        if self._points is None:
            self._points = np.array(list(map(lambda x: x.data, self.inliers)), dtype=float).reshape(len(self.inliers) * 2, 2)
        return self._points

    @property
    def direction(self):
        return self.model[:2] / self.model[2]

    def get_inliers(self, lines, angle_threshold=np.radians(3)):
        
        locations = np.array(list(map(lambda x: x.midpoint, lines)))
        directions = np.array(list(map(lambda x: x.direction, lines)))
        #directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

        cosine_thetas = angle_with_vp(self.model, locations, directions)

        theta_thresh = np.cos(angle_threshold)
        indices = np.argwhere(cosine_thetas > theta_thresh).flatten()
        
        return [lines[i] for i in indices]

    def is_inlier(self, line, angle_threshold=np.radians(3)):
        return len(self.get_inliers([line], angle_threshold)) > 0
        
class Edglets:
    def __init__(self, locations, directions, strengths):
        self.locations = locations
        self.directions = directions
        self.strengths = strengths

        self.normals = np.zeros_like(self.directions)
        self.normals[:, 0] = self.directions[:, 1]
        self.normals[:, 1] = -self.directions[:, 0]
        p = -np.sum(self.locations * self.normals, axis=1)

        self.lines = np.concatenate((self.normals, p[:, np.newaxis]), axis=1)

class Direction(IntEnum):
    Vertical = 0
    Horizontal = 1

class VanishingPointFinder():

    def __init__(self, lines, direction:Direction=None, angle_threshold=np.radians(80)):
        super().__init__()

        self.lines = lines
        self.direction = direction
        self.angle_threshold = angle_threshold

    def compute_edgelets(self):

        if len(self.lines) < 2: return None

        locations = []
        directions = []
        strengths = []

        for line in self.lines:
            locations.append(line.midpoint)
            directions.append(line.direction)
            strengths.append(line.length)

        locations = np.array(locations)
        strengths = np.array(strengths)
        directions = np.array(directions)
        #directions = np.array(directions) / np.linalg.norm(directions, axis=1)[:, np.newaxis]

        return Edglets(locations, directions, strengths)

    def compute_votes(self, model, threshold_inlier):

        cosine_theta = angle_with_vp(model, self.edgelets.locations, self.edgelets.directions)
        theta_thresh = np.cos(threshold_inlier)

        return (cosine_theta > theta_thresh) * self.edgelets.strengths

    def solve(self, max_iterations=2000, threshold_inlier=math.radians(2), max_time=0.33):

        self.edgelets = self.compute_edgelets()

        if self.edgelets is None:
            return []

        num_pts = self.edgelets.strengths.size

        arg_sort = np.argsort(-self.edgelets.strengths)
        first_index_space = arg_sort[:num_pts // min(5, len(self.edgelets.lines))]
        second_index_space = arg_sort[:num_pts // 2]

        self.best_model = None
        self.vanishing_points = []
        
        max_iterations = min(max_iterations, len(self.lines) * 40)
        start_time = time.time() 

        for ransac_iter in range(max_iterations):
            if time.time() - start_time > max_time:
                break

            ind1 = np.random.choice(first_index_space)
            ind2 = np.random.choice(second_index_space)

            l1 = self.edgelets.lines[ind1]
            l2 = self.edgelets.lines[ind2]

            current_model = np.cross(l1, l2)

            if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
                # reject degenerate candidates
                continue

            # if current_model[1] / current_model[2] > 1000:
            #     continue

            if self.direction is not None:
                line1 = self.lines[ind1]
                line2 = self.lines[ind2]

                if self.direction == Direction.Vertical:
                    if LineFunctions.line_angle_difference(line1.angle, pi_2) > self.angle_threshold or LineFunctions.line_angle_difference(line2.angle, pi_2) > self.angle_threshold:
                        continue
                else:
                    if LineFunctions.line_angle_difference(line1.angle, 0) > self.angle_threshold or LineFunctions.line_angle_difference(line2.angle, 0) > self.angle_threshold:
                        continue


            current_model = current_model / current_model[2]

            vp = VanishingPoint(self.lines, current_model, self.compute_votes(current_model, threshold_inlier))
            
            self.vanishing_points.append(vp)

        self.vanishing_points.sort(key=lambda x:x.score, reverse=True)

        return self.vanishing_points

class VpfProcess(BaseProcess):
    def solve(self, lines, merge_width=0.0, threshold_inlier=math.radians(3), max_iterations=2000, max_time=0.333):

        if merge_width > 0.0:
            lines = merge_lines(lines, search_width=merge_width, search_length=1.1)

        return VanishingPointFinder(lines).solve(threshold_inlier=threshold_inlier, max_iterations=max_iterations, max_time=max_time)

class PipelineVanishingPointFinder(PipelineStep):
    def __init__(self, pipeline):
        super().__init__(pipeline)

        self.mp = Multiprocessor(VpfProcess)
        self.mp.start()

    async def stop(self):
        self.mp.stop()
        await super().stop()

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.VanishingPoints

    @property
    def required_keys(self) -> list:
        return ["room", "downscaled", "isolated", "lines"]

    @property
    def output_keys(self) -> list:
        return []

    def get_contour_lines(self, image, surface, min_length):
        lines = []

        for poly in surface.polygons:
            num_pts = len(poly)

            for i in range(num_pts):
                point_a = poly[i][0]
                point_b = poly[(i+1) % num_pts][0]

                if line_on_image_edge(point_a, point_b, image.shape[1], image.shape[0]):
                    continue

                if distance.euclidean(point_a, point_b) > min_length:
                    lines.append(Line(point_b[0], point_b[1], point_a[0], point_a[1]))
        
        return lines
                

    def run(self, data):

        room = data["room"]
        image = data["downscaled"]
        diagonal = math.hypot(image.shape[0], image.shape[1])

        surfaces = []

        surfaces.extend(data["room"].get_surfaces(surfaceTypes=[SurfaceType.Floor, SurfaceType.Wall]))
        surfaces.extend(data["room"].get_surfaces(labels=box_like))
        #self.surfaces.extend(data["room"].get_surfaces(labels=legged_objects))

        #find single vertical vanishing point
        vertical_threshold = np.radians(15)
        vertical_lines = []
        all_lines = []

        vpfs = []

        for surface in surfaces:
            lines = surface.lines.copy()
            lines.extend(self.get_contour_lines(image, surface, diagonal/80))
            all_lines.extend(lines)
   
            if surface.surfaceType == SurfaceType.Wall or surface.bestLabel in box_like:                
                vertical, horizontal = partition(lambda x: LineFunctions.line_angle_difference(x.angle, pi_2) < vertical_threshold, lines)
                vertical_lines.extend(vertical)
                self.mp.schedule('solve', horizontal, diagonal/200.0)
            else:
                self.mp.schedule('solve', lines, 0, math.radians(5), 500, 0.2)

        #find vertical vanishing point for entire room
        if len(vertical_lines) > 1:
            self.mp.schedule('solve', vertical_lines, diagonal/200.0, math.radians(3), 2000, 0.5)

        results = self.mp.await_completion()

        room.vertical_vp = results[-1]

        if len(room.vertical_vp) == 0:
            vpf = VanishingPointFinder(vertical_lines)
            room.vertical_vp = vpf.solve(threshold_inlier=np.radians(5))

        for i in range(len(surfaces)):
            surface = surfaces[i]
            vp = results[i]
            if surface.surfaceType == SurfaceType.Wall or surface.bestLabel in box_like:
                surface.horizontal_vp = vp
            else:
                surface.vp = vp

            if surface.vertical_vp is None:
                surface.vertical_vp = room.vertical_vp

        if im_logging_enabled(data):
            log_image(data, "vanishing_points", self.get_debug_image(data, surfaces, all_lines))

    def get_debug_image(self, data, surfaces, all_lines):
           
        room = data["room"]
        image = data["downscaled"].copy()
    
        #draw all lines
        draw_lines(image, all_lines, color=(80,80,80))

        if room.vertical_vp is not None and len(room.vertical_vp) > 0:
            draw_lines(image, room.vertical_vp[0].inliers, color=(0,255,0), thickness=2)

        for surface in surfaces:
            if surface.horizontal_vp is not None and len(surface.horizontal_vp) > 0:
                draw_lines(image, surface.horizontal_vp[0].inliers, color=random_color(), thickness=2)

            if surface.vp and len(surface.vp):
                draw_lines(image, surface.vp[0].inliers, color=random_color(), thickness=2)
            
        return image

