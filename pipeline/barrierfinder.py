import numpy as np
from scipy import ndimage
import cv2
import random
import time
from scipy.spatial import distance
import math

from .ade20k import ADE20K
from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .utils import resize_array, random_color, overlay_mask, normalize, convert_color, put_text, adjust_mask
from .planegeometry import Dimension
from .extractsurfaces import box_like, legged_objects
from .vanishingpointfinder import angle_with_vp
from .logging import log_image, log_segmentation_image, im_logging_enabled
from cambrian.LineFunctions import LineFunctions
from .Line import line_angle_difference, Line, line_on_image_edge
from .room import Room, Surface

def closest_polygon_side(contour, point, min_length_threshold=None, max_length_threshold=None):
    #return (distance, point, and indices) of closest line in polygon or contour by midpoints

    num_pts = len(contour)
    min_dist_sq = np.inf
    min_index = None

    min_length_threshold_sq = None if min_length_threshold is None else min_length_threshold * min_length_threshold
    max_length_threshold_sq = None if max_length_threshold is None else max_length_threshold * max_length_threshold

    for i in range(num_pts):
        point_a = contour[i][0]
        point_b = contour[(i+1) % num_pts][0]

        if point_a[0] == point_b[0] and point_a[1] == point_b[1]: continue

        if min_length_threshold_sq is not None or max_length_threshold_sq is not None:
            length_sq = distance.sqeuclidean(point_a, point_b)

            if min_length_threshold_sq is not None and length_sq < min_length_threshold_sq: continue
            if max_length_threshold_sq is not None and length_sq > max_length_threshold_sq: continue

        midpoint = (point_a[0] + point_b[0]) / 2, (point_a[1] + point_b[1]) / 2

        #midpoints between point a and midpoint
        point_a = (point_a[0] + midpoint[0]) / 2, (point_a[1] + midpoint[1]) / 2
        point_b = (point_b[0] + midpoint[0]) / 2, (point_b[1] + midpoint[1]) / 2

        dist_point_a_sq = distance.sqeuclidean(point_a, point)
        dist_point_b_sq = distance.sqeuclidean(point_b, point)
        dist_midpoint_sq = distance.sqeuclidean(midpoint, point)

        if dist_point_a_sq < min_dist_sq:
            min_index = i
            min_dist_sq = dist_point_a_sq

        if dist_point_b_sq < min_dist_sq:
            min_index = i
            min_dist_sq = dist_point_b_sq

        if dist_midpoint_sq < min_dist_sq:
            min_index = i
            min_dist_sq = dist_midpoint_sq

    if min_index is None:
        if min_length_threshold is not None:
            return closest_polygon_side(contour, point, min_length_threshold=None, max_length_threshold=max_length_threshold)
        elif max_length_threshold is not None:
            return closest_polygon_side(contour, point, min_length_threshold=None, max_length_threshold=None)
        
    return min_index, np.sqrt(min_dist_sq)

class Barrier():
    def __init__(self, surface_barrier, line, vanishing_point, angle_threshold):
        self.surface_barrier = surface_barrier
        self.line = line
        self.vanishing_point = vanishing_point
        self.shape_line = None
        self.closest_point = None

        min_dist = np.inf
        min_shape = None
        min_index = None

        for i in range(len(self.surface_barrier.shapes)):
            shape = self.surface_barrier.shapes[i]
            index, dist = closest_polygon_side(shape, self.line.midpoint)

            if dist < min_dist:
                min_dist = dist
                min_index = index
                min_shape = shape
        
        if min_shape is not None:
            #now check these indices, and indices before and after, for the most like line by angle:
            num_pts = len(min_shape)

            a = min_index - 1
            if a < 0: a = num_pts - 1
            b = min_index
            c = (min_index + 1) % num_pts
            d = (min_index + 2) % num_pts

            point_a = min_shape[a][0]
            point_b = min_shape[b][0]
            point_c = min_shape[c][0]
            point_d = min_shape[d][0]

            ab_angle = LineFunctions.line_angle_difference(LineFunctions.line_angle(point_a[0], point_a[1], point_b[0], point_b[1]), line.angle)
            bc_angle = LineFunctions.line_angle_difference(LineFunctions.line_angle(point_b[0], point_b[1], point_c[0], point_c[1]), line.angle)
            cd_angle = LineFunctions.line_angle_difference(LineFunctions.line_angle(point_c[0], point_c[1], point_d[0], point_d[1]), line.angle)

            best_angle = ab_angle
            best_points = point_a, point_b

            if bc_angle < best_angle:
                best_angle = bc_angle
                best_points = point_b, point_c

            if cd_angle < best_angle:
                best_angle = cd_angle
                best_points = point_c, point_d
            
            self.shape_line = Line(np.array([best_points[0][0], best_points[0][1], best_points[1][0], best_points[1][1]]))
            self.closest_point = self.shape_line.closest_point(self.line.midpoint)

    def debug(self, img, color):
        self.line.draw(img, color=color, thickness=2)
        point_a = (int(self.line.midpoint[0]), int(self.line.midpoint[1]))
        point_b = (int(self.closest_point[0]), int(self.closest_point[1]))
        
        cv2.line(img, point_a, point_b, color, 1)
        cv2.line(img, (int(self.shape_line.point_a[0]), int(self.shape_line.point_a[1])),(int(self.shape_line.point_b[0]), int(self.shape_line.point_b[1])), color, 1)

class SurfaceBarriers():
    def __init__(self, data, surface, vanishing_points):
        self.data = data
        self.surface = surface
        self.vanishing_points = vanishing_points
        self.image = self.data["downscaled"]
        self.diagonal = math.hypot(self.image.shape[0], self.image.shape[1])

        self.barrier_candidates = self.get_barrier_candidates()

    def get_barrier_candidates(self, angle_threshold=np.radians(45)):

        def inside_mask(mask, point):
            if point[0] < mask.shape[1] and point[1] < mask.shape[0]:
                return mask[int(point[1]), int(point[0])] > 0
            return False

        
        padding = 10
        surface_mask = cv2.copyMakeBorder(self.surface.mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0) 
        trans = cv2.distanceTransform(1-surface_mask, cv2.DIST_L2, 5)
        _, outer_mask = cv2.threshold(trans, 0.05 * trans.max(), 1, 0)

        trans = cv2.distanceTransform(surface_mask, cv2.DIST_L2, 5)
        _, inner_mask = cv2.threshold(trans, 0.5 * trans.max(), 1, 0)

        trans = cv2.distanceTransform(surface_mask, cv2.DIST_L2, 5)
        _, shape_mask = cv2.threshold(trans, 0.1 * trans.max(), 1, 0)
        shape_mask = shape_mask[padding:-padding,padding:-padding].astype(np.uint8)
        contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.shapes = list(map(lambda contour: cv2.approxPolyDP(contour, 0.003 * cv2.arcLength(contour, True), True), contours))

        self.mask_edges = 1 - inner_mask - outer_mask
        self.mask_edges[self.mask_edges < 0] = 0
        self.mask_edges = self.mask_edges[padding:-padding,padding:-padding]

        self.candidates = []

        for line in self.data["lines"]:

            if line_on_image_edge(line.point_a, line.point_b, self.image):
                continue

            point_a = (line.point_a[0] + line.midpoint[0]) / 2, (line.point_a[1] + line.midpoint[1]) / 2
            point_b = (line.point_b[0] + line.midpoint[0]) / 2, (line.point_b[1] + line.midpoint[1]) / 2

            if inside_mask(self.mask_edges, line.midpoint) and (inside_mask(self.mask_edges, point_a) or inside_mask(self.mask_edges, point_b)):
                self.candidates.append(line)

        self.candidates = Line.merge(self.candidates, search_width=self.diagonal/200, search_length=1.2)          

        for poly in self.surface.polygons:
            num_pts = len(poly)
            for i in range(num_pts):
                point_a = poly[i][0]
                point_b = poly[(i+1) % num_pts][0]

                if line_on_image_edge(point_a, point_b, self.image):
                    continue

                line = Line(np.array([point_a[0], point_a[1], point_b[0], point_b[1]]))
                self.candidates.append(line)

        self.candidates = Line.merge(self.candidates, search_width=self.diagonal/400, search_length=1.0)

        #filter and correct to vp
        barriers = []
        for line in self.candidates:
        
            #check for validity with center: (todo: use closest polygonal point)
            # midpoint_angle = LineFunctions.line_angle(self.surface.center[0], self.surface.center[1], line.midpoint[0], line.midpoint[1])
            # if line_angle_difference(midpoint_angle, line.angle) < np.radians(45):
            #     continue

            #check for validity with vanishing point:
            vp_match = next(filter(lambda vp: vp.is_inlier(line, np.radians(5)), self.vanishing_points), None)

            if vp_match is None:
                continue

            barrier = Barrier(self, line, vp_match, angle_threshold)

            # #should be fairly perpendicular:
            #barrier_angle = LineFunctions.line_angle(barrier.closest_point[0], barrier.closest_point[1], line.midpoint[0], line.midpoint[1])
            if line_angle_difference(barrier.shape_line.angle, line.angle) < angle_threshold:
                barriers.append(barrier)

        return barriers

    def debug(self, img, color):
        for shape in self.shapes:
            cv2.drawContours(img, [shape], -1, color=(255,255,255), thickness=1)

        Line.draw_all(img, self.candidates, color=(0,0,0), thickness=1)

        for barrier in self.barrier_candidates:
            barrier.debug(img, color=color)
        

class BarrierSolver():
    def __init__(self, data, surface_barriers):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface_barriers = surface_barriers


    def solve(self, max_iterations=2000, threshold_inlier=math.radians(2), max_time=0.33, measure_area=False):     

        max_iterations = min(max_iterations, len(self.surface_barriers) * 40)
        start_time = time.time() 

        num_samples = random.randint(2, len(self.surface_barriers))
        for ransac_iter in range(max_iterations):
            if time.time() - start_time > max_time:
                break
        
class PipelineBarrierFinder(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Barriers

    @property
    def required_keys(self) -> list:
        return ["room", "downscaled", "isolated", "lines"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):

        self.data = data
        self.image = self.data["downscaled"]
        self.room = self.data["room"]
        self.diagonal = math.hypot(self.image.shape[0], self.image.shape[1])

        self.surfaces = []
        self.surfaces.extend(self.room.get_surfaces(surfaceTypes=[SurfaceType.Wall, SurfaceType.WallLike]))
        self.surfaces.extend(self.room.get_surfaces(labels=box_like))

        self.vanishing_points = []
        for surface in self.surfaces:
            for vp in surface.vanishing_points:
                if vp not in self.vanishing_points:
                    self.vanishing_points.append(vp)

        self.lines = []

        self.barriers = {}

        for surface in self.surfaces:
            self.barriers[surface.uniqueId] = SurfaceBarriers(self.data, surface, self.vanishing_points)

        if im_logging_enabled(self.data):
            log_image(self.data, "potential_barriers.png", self.get_debug_image())

        bs = BarrierSolver(self.data, self.barriers)
        bs.solve()


    def get_debug_image(self):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.room.surfaces))

        #overlay probs

        for i in range(len(self.room.surfaces)):
            surface = self.room.surfaces[i]
            
            #mask = (self.barriers[surface.uniqueId][1] if surface.uniqueId in self.barriers else surface.mask) > 0
            mask = surface.mask > 0
            
            max_value = 0.9
            if max_value > 0:
                img_hsv[:, :, 0][mask] = hues[i]
                img_hsv[:, :, 1][mask] = 255 * np.power(surface.probs[mask], 0.15)
                    
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)

        for i in range(len(self.room.surfaces)):
            surface = self.room.surfaces[i]
            color = convert_color((hues[i],127,255), cv2.COLOR_HSV2RGB_FULL)

            if surface.uniqueId in self.barriers:
                self.barriers[surface.uniqueId].debug(img, color)

        return img
