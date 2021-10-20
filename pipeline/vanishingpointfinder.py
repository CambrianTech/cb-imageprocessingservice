import numpy as np
from scipy import ndimage
import cv2
import math
import time
from enum import IntEnum

from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .utils import resize_array, random_color, overlay_mask
from .planegeometry import Dimension
from .logging import log_image, log_segmentation_image, im_logging_enabled
from .Line import Line, line_angle_difference
from .room import Room, Surface

class VanishingPoint:
    def __init__(self, model, votes):

        self.model = model
        self.votes = votes
        self._score = sum(self.votes)

    def __eq__(self, other):
        return self.score() == other.score()

    def __lt__(self, other):
        return self.score() < other.score()

    @property
    def score(self):
        return self._score

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

    def __init__(self, data, surface, direction:Direction=None, edgelets=None):
        super().__init__()
        self.data = data
        self.surface = surface
        self.direction = direction
        self.edgelets = edgelets

    def compute_edgelets(self):

        if len(self.surface.lines) < 2: return None

        locations = []
        directions = []
        strengths = []

        for line in self.surface.lines:
            p0, p1 = np.array([line.point_a[0], line.point_a[1]]), np.array([line.point_b[0], line.point_b[1]])

            locations.append(line.midpoint)
            directions.append(p1 - p0)
            strengths.append(line.length)


        locations = np.array(locations)
        directions = np.array(directions)
        strengths = np.array(strengths)
        directions = np.array(directions) / np.linalg.norm(directions, axis=1)[:, np.newaxis]

        return Edglets(locations, directions, strengths)

    def solve(self, num_ransac_iter=2000, threshold_inlier=math.radians(7), max_time=1.0):

        if self.edgelets is None:
            self.edgelets = self.compute_edgelets()

        if self.edgelets is None:
            return []

        num_pts = self.edgelets.strengths.size

        arg_sort = np.argsort(-self.edgelets.strengths)
        first_index_space = arg_sort[:num_pts // min(5, len(self.edgelets.lines))]
        second_index_space = arg_sort[:num_pts // 2]

        self.best_model = None
        self.vanishing_points = []
        t = time.time()

        threshold_horizontal = np.radians(75)

        for ransac_iter in range(num_ransac_iter):
            if time.time() - t > max_time:
                return self.best_model

            ind1 = np.random.choice(first_index_space)

            ind2 = np.random.choice(second_index_space)

            l1 = self.edgelets.lines[ind1]
            l2 = self.edgelets.lines[ind2]

            line1 = self.surface.lines[ind1]
            line2 = self.surface.lines[ind2]

            current_model = np.cross(l1, l2)

            if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
                # reject degenerate candidates
                continue


            if self.direction is not None:

                both_consistent = (current_model[1] / current_model[2] > 1000)

                if self.direction == Direction.Vertical:
                    vdt1 = abs(np.dot(self.edgelets.directions[ind1], [0, 1]))
                    vdt2 = abs(np.dot(self.edgelets.directions[ind2], [0, 1]))

                    if vdt1 < .95 or vdt2 < .95 or not both_consistent:
                        continue
                else:
                    # hdt1 = abs(np.dot(self.edgelets.directions[ind1], [1, 0]))
                    # hdt2 = abs(np.dot(self.edgelets.directions[ind2], [1, 0]))

                    if line_angle_difference(line1.angle, 0) > threshold_horizontal or line_angle_difference(line2.angle, 0) > threshold_horizontal:
                        continue


            current_model = current_model / current_model[2]

            vp = VanishingPoint(current_model, self.compute_votes(current_model, threshold_inlier))
            
            self.vanishing_points.append(vp)

        self.vanishing_points.sort(key=lambda x:x.score, reverse=True)

        return self.vanishing_points


    def compute_votes(self, model, threshold_inlier):

        vp = model[:2] / model[2]

        est_directions = self.edgelets.locations - vp

        dot_prod = np.sum(est_directions * self.edgelets.directions, axis=1)
        abs_prod = np.linalg.norm(self.edgelets.directions, axis=1) * \
                   np.linalg.norm(est_directions, axis=1)
        abs_prod[abs_prod == 0] = 1e-5

        cosine_theta = np.abs(dot_prod / abs_prod)

        theta_thresh = np.cos(threshold_inlier)

        return (cosine_theta > theta_thresh) * self.edgelets.strengths



class PipelineVanishingPointFinder(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.VanishingPoints

    @property
    def required_keys(self) -> list:
        return ["room", "downscaled", "isolated", "lines"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):

        self.surfaces = []

        self.surfaces.extend(data["room"].get_surfaces(SurfaceType.Wall))

        for surface in self.surfaces:

            vpf = VanishingPointFinder(data, surface, direction=Direction.Horizontal)
            surface.horizontal_vp = vpf.solve()

            vpf = VanishingPointFinder(data, surface, direction=Direction.Vertical, edgelets=vpf.edgelets)
            surface.vertical_vp = vpf.solve()

        if im_logging_enabled(data):
            log_image(data, "vanishing_points", self.get_debug_image(data))

    def get_debug_image(self, data):
           
        img = data["downscaled"].copy()

        for surface in self.surfaces:

            #draw all lines
            for line in surface.lines:
                line.draw(img, color=(80,80,80))

            def draw_vp(vp):
                inliers = np.array(surface.lines)[vp.votes > 0]
                color = random_color()

                for line_data in inliers:
                    line = Line(line_data)
                    line.draw(img, color=color)
                
            if len(surface.horizontal_vp) > 0:
                draw_vp(surface.horizontal_vp[0])

            if len(surface.vertical_vp) > 0:
                draw_vp(surface.vertical_vp[0])
            
        return img

