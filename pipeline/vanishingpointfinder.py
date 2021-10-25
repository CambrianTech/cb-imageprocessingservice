import numpy as np
from scipy import ndimage
import cv2
import math
import time
from enum import IntEnum

from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .utils import resize_array, random_color, overlay_mask, partition
from .planegeometry import Dimension
from .extractsurfaces import box_like
from .logging import log_image, log_segmentation_image, im_logging_enabled
from .Line import Line, line_angle_difference, on_image_edge
from .room import Room, Surface
from .ade20k import ADE20K

find_horizontal = True

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
    def __init__(self, lines, model, votes, measure_area=False):

        self.lines = lines
        self.model = model
        self.votes = votes
        self.measure_area = measure_area

        self._score = None
        self._inliers = None

        #cv2.minAreaRect(InputArray  points)

    def __eq__(self, other):
        return self.score() == other.score()

    def __lt__(self, other):
        return self.score() < other.score()

    @property
    def score(self):
        if self._score is None:
            self._score = sum(self.votes)

            if self.measure_area and len(self.inliers) > 1:
                all_points = self.inliers.reshape((self.inliers.shape[0] * 2, 2)).astype(int)
                rect = cv2.minAreaRect(all_points)
                self._score = self._score * np.hypot(rect[1][0], rect[1][1])

        return self._score

    @property
    def inliers(self):
        if self._inliers is None:
            self._inliers = np.array(self.lines)[self.votes > 0]

        return self._inliers

    @property
    def direction(self):
        return self.model[:2] / self.model[2]
        
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
        directions = np.array(directions) / np.linalg.norm(directions, axis=1)[:, np.newaxis]

        return Edglets(locations, directions, strengths)

    def compute_votes(self, model, threshold_inlier):

        cosine_theta = angle_with_vp(model, self.edgelets.locations, self.edgelets.directions)
        theta_thresh = np.cos(threshold_inlier)

        return (cosine_theta > theta_thresh) * self.edgelets.strengths

    def solve(self, max_iterations=1000, threshold_inlier=math.radians(3), max_time=0.25, measure_area=False):

        self.edgelets = self.compute_edgelets()

        if self.edgelets is None:
            return []

        num_pts = self.edgelets.strengths.size

        arg_sort = np.argsort(-self.edgelets.strengths)
        first_index_space = arg_sort[:num_pts // min(5, len(self.edgelets.lines))]
        second_index_space = arg_sort[:num_pts // 2]

        self.best_model = None
        self.vanishing_points = []
        
        pi_2 = np.pi/2       

        max_iterations = min(max_iterations, len(self.lines) * 30)
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
                    if line_angle_difference(line1.angle, pi_2) > self.angle_threshold or line_angle_difference(line2.angle, pi_2) > self.angle_threshold:
                        continue
                else:
                    if line_angle_difference(line1.angle, 0) > self.angle_threshold or line_angle_difference(line2.angle, 0) > self.angle_threshold:
                        continue


            current_model = current_model / current_model[2]

            vp = VanishingPoint(self.lines, current_model, self.compute_votes(current_model, threshold_inlier), measure_area=measure_area)
            
            self.vanishing_points.append(vp)

        self.vanishing_points.sort(key=lambda x:x.score, reverse=True)

        return self.vanishing_points

def draw_vp(img, vp, color, thickness=1):
    for line_data in vp.inliers:
        line = Line(line_data)
        line.draw(img, color=color, thickness=thickness)

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

    def get_contour_lines(self, surface, min_length=None):
        lines = []

        if min_length is None:
            min_length = self.diagonal/80

        for poly in surface.polygons:
            num_pts = len(poly)

            for i in range(num_pts):
                point_a = poly[i][0]
                point_b = poly[(i+1) % num_pts][0]

                if on_image_edge(point_a, self.image) and on_image_edge(point_b, self.image):
                    continue

                line = Line(np.array([(point_b[0], point_b[1], point_a[0], point_a[1])], dtype=np.int).reshape(4), group="contour")
                if line.length > min_length:
                    lines.append(line)
        
        return lines
                

    def run(self, data):

        self.image = data["downscaled"]
        self.diagonal = math.hypot(self.image.shape[0], self.image.shape[1])

        self.surfaces = []

        self.surfaces.extend(data["room"].get_surfaces(surfaceTypes=[SurfaceType.Floor, SurfaceType.Wall, SurfaceType.WallLike]))
        self.surfaces.extend(data["room"].get_surfaces(labels=box_like))


        #find single vertical vanishing point
        pi_2 = np.pi/2
        vertical_threshold = np.radians(15)
        vertical_lines = []
        self.all_lines = []

        for surface in self.surfaces:
            lines = surface.lines.copy()
            lines.extend(self.get_contour_lines(surface))
            self.all_lines.extend(lines)

            if surface.surfaceType == SurfaceType.Floor or surface.surfaceType == SurfaceType.Ceiling:
                #used for legs and objects setting upon:
                vpf = VanishingPointFinder(lines)
                surface.vp = vpf.solve()
            else:                
                vertical, horizontal = partition(lambda x: line_angle_difference(x.angle, pi_2) < vertical_threshold, lines)
                horizontal = Line.merge(horizontal, search_width=self.diagonal/200, search_length=1.1)

                vpf = VanishingPointFinder(horizontal)
                surface.horizontal_vp = vpf.solve(measure_area=True)

                vertical_lines.extend(vertical)

        #find vertical vanishing point for entire room
        if len(vertical_lines) > 1:
            vertical_lines = Line.merge(vertical_lines, search_width=self.diagonal/200, angle_threshold=math.radians(5))
            vpf = VanishingPointFinder(vertical_lines)
            self.vertical_vp = vpf.solve(threshold_inlier=np.radians(3), max_time=0.5)
            if len(self.vertical_vp) == 0:
                self.vertical_vp = vpf.solve(threshold_inlier=np.radians(20))


        for surface in self.surfaces:
            if surface.vertical_vp is None:
                surface.vertical_vp = self.vertical_vp


        if im_logging_enabled(data):
            log_image(data, "vanishing_points", self.get_debug_image(data))

    def get_debug_image(self, data):
           
        img = self.image.copy()
    
        #draw all lines
        Line.draw_all(img, self.all_lines, color=(80,80,80))

        if self.vertical_vp is not None and len(self.vertical_vp) > 0:
            draw_vp(img, self.vertical_vp[0], color=(0,255,0))

        for surface in self.surfaces:
            if surface.horizontal_vp is not None and len(surface.horizontal_vp) > 0:
                draw_vp(img, surface.horizontal_vp[0], color=random_color())

            if surface.vp and len(surface.vp):
                draw_vp(img, surface.vp[0], color=random_color())
            
        return img

