import numpy as np
from scipy import ndimage
import cv2
import random
import time
from scipy.spatial import distance
import math

from .ade20k import ADE20K
from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .utils import resize_array, random_color, overlay_mask, normalize, convert_color, put_text, adjust_mask, partition
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
    def __init__(self, surface_barrier, line, vanishing_point):
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

        #get neighbor, if any
        self.surface_neighbor = None

        def get_neighbor_points(normal, num_pts=None):
            max_distance = min(self.line.length / 2, self.surface_barrier.diagonal / 50)

            points = []
            distances = range(int(max_distance / 2), int(max_distance), int(max_distance / 5)) if max_distance > 10 else [max_distance]

            for distance in distances:
                line_points = self.line.get_points(self.surface_barrier.image.shape[1], self.surface_barrier.image.shape[0], num_pts)
                
                if len(line_points) > 2:
                    line_points = line_points[1:-1]

                for line_point in line_points:
                    test_point = normal * distance + np.array(line_point)
                    points.append(test_point)

            return points

        def get_potential_neighbor(normal, num_pts=None):
            points = get_neighbor_points(normal, num_pts)
            surface = None

            for point in points:
                surface = self.surface_barrier.data["room"].surface_at_point(point)
                if surface != self.surface_barrier.surface:
                    return surface
            return surface

        test_surface_a = get_potential_neighbor(self.line.normal_a, 7)
        test_surface_b = get_potential_neighbor(self.line.normal_b, 7)

        if test_surface_a != test_surface_b:
            self.surface_neighbor = test_surface_a if test_surface_a is not None else test_surface_b

    def debug(self, img, color):

        if self.surface_neighbor is not None:
            self.line.draw(img, color=color, thickness=3)
        else:
            self.line.draw(img, color=color, thickness=1)

        point_a = (int(self.line.midpoint[0]), int(self.line.midpoint[1]))
        point_b = (int(self.closest_point[0]), int(self.closest_point[1]))
        
        cv2.line(img, point_a, point_b, color, 1)
        cv2.line(img, (int(self.shape_line.point_a[0]), int(self.shape_line.point_a[1])),(int(self.shape_line.point_b[0]), int(self.shape_line.point_b[1])), color, 1)

class BarrierGroup():
    def __init__(self, barrier):
        self.barriers = [barrier]

    def add_barrier(self, barrier):
        self.barriers.append(barrier)

    def match_score(self, test_barrier, min_angle_diff, search_width, length_multiplier):

        test_rect = test_barrier.line.bounding_box(search_width, length_multiplier)

        score = 0

        for barrier in self.barriers:

            angle = LineFunctions.line_angle_difference(barrier.line.angle, test_barrier.line.angle)

            if angle > min_angle_diff: 
                continue

            if test_barrier.vanishing_point != barrier.vanishing_point:
                continue

            if test_barrier.surface_neighbor != barrier.surface_neighbor and test_barrier.surface_neighbor != None and barrier.surface_neighbor != None:
                continue

            # length_ratio = test_barrier.line.length / barrier.line.length
            # length_ratio = min(length_ratio, 1/length_ratio)

            # if length_ratio < 0.1:
            #     continue

            rect = barrier.line.bounding_box(search_width, length_multiplier)

            result, region = cv2.rotatedRectangleIntersection(rect, test_rect)

            if result != 0:
                score += 1

        return score

    @property
    def surface_neighbor(self):
        #return first one: they must all be the same for all my barriers, see filtering in self.match_score
        for barrier in self.barriers:
            if barrier.surface_neighbor is not None:
                return barrier.surface_neighbor

        return None

    def debug(self, img, color):

        points = []
        marker_color = random_color()
        for barrier in self.barriers:
            points.append(barrier.line.point_a)
            points.append(barrier.line.point_b)

            #cv2.drawMarker(img, (int(barrier.line.midpoint[0]), int(barrier.line.midpoint[1])), marker_color, thickness=2)

        rect = cv2.minAreaRect(np.array(points))
        rect_width = min(rect[1][0], rect[1][1])

        box = cv2.boxPoints(rect)
        box = np.int0(box)
        cv2.drawContours(img, [box], 0, (255,0,0), 1)
            

class SurfaceBarriers():
    def __init__(self, data, surface, vanishing_points):
        self.data = data
        self.surface = surface
        self.vanishing_points = vanishing_points
        self.image = self.data["downscaled"]
        self.room = self.data["room"]
        self.diagonal = math.hypot(self.image.shape[0], self.image.shape[1])

        self.barrier_candidates = self.get_barrier_candidates()
        self.barrier_groups = self.group_barriers()

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
            elif self.surface.parent is not None:
                print('%s: Search within parent %s' % (self.surface.name, self.surface.parent.name))

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

            barrier = Barrier(self, line, vp_match)

            # #should be fairly perpendicular:
            #barrier_angle = LineFunctions.line_angle(barrier.closest_point[0], barrier.closest_point[1], line.midpoint[0], line.midpoint[1])
            if line_angle_difference(barrier.shape_line.angle, line.angle) < angle_threshold:
                barriers.append(barrier)

        return barriers

    def group_barriers(self, min_angle_diff=np.radians(10)):
        
        barrier_groups = []

        def get_best_match(search_width, length_multiplier):
            best_match = None
            best_score = 0

            for test_group in barrier_groups:

                score = test_group.match_score(barrier, min_angle_diff=min_angle_diff, search_width=search_width, length_multiplier=length_multiplier)

                if score > best_score:
                    best_match = test_group
                    best_score = score

            return best_match

        search_width=self.diagonal/40
        length_multiplier=0.7

        #group width-wise
        orphaned = []
        for barrier in self.barrier_candidates:

            best_match = get_best_match(search_width=search_width, length_multiplier=length_multiplier)
                    
            if best_match is not None:
                best_match.add_barrier(barrier)
            elif barrier.surface_neighbor is not None:
                barrier_groups.append(BarrierGroup(barrier))
            else:
                orphaned.append(barrier)

        #retry ones without siblings, wider field
        barrier_groups, poor_barrier_groups = partition(lambda x: len(x.barriers) > 1, barrier_groups)
        
        for group in poor_barrier_groups:
            orphaned.extend(group.barriers)        

        for barrier in orphaned:

            best_match = get_best_match(search_width=search_width, length_multiplier=length_multiplier)

            if best_match is not None:
                best_match.add_barrier(barrier)
            elif barrier.surface_neighbor is not None:
                barrier_groups.append(BarrierGroup(barrier))

        return barrier_groups


    def debug(self, img, color):
        for shape in self.shapes:
            cv2.drawContours(img, [shape], -1, color=(255,255,255), thickness=1)

        Line.draw_all(img, self.candidates, color=(0,0,0), thickness=1)

        for barrier in self.barrier_candidates:
            barrier.debug(img, color=color)

        for barrier_group in self.barrier_groups:
            barrier_group.debug(img, color=color)
        

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
