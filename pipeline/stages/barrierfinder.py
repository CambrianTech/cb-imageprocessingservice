import numpy as np
from scipy import ndimage
import cv2
import random
import time
from enum import IntEnum
from scipy.spatial import distance
import math
from skimage.segmentation import watershed

from pipeline.data.ade20k import ADE20K
from pipeline.data.surface_type import SurfaceType
from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.misc.utils import resize_array, random_color, overlay_mask, normalize, convert_color, put_text, adjust_mask, partition
from .planegeometry import Dimension
from .extractsurfaces import box_like, legged_objects
from .vanishingpointfinder import angle_with_vp
from pipeline.data.logging import log_image, log_segmentation_image, im_logging_enabled
from cambrian.LineFunctions import LineFunctions
from pipeline.components.line import line_angle_difference, Line, line_on_image_edge
from pipeline.components.rotated_rect import RotatedRect
from pipeline.components.room import Room
from pipeline.components.surface import Surface

debug_indices = [147]
debug_show_indices = True

debug_objects = []

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

        if test_surface_a is not None or test_surface_b is not None:
            self.surface_neighbor = test_surface_a if test_surface_a is not None else test_surface_b

    def debug(self, img, color):

        if self.surface_neighbor is not None:
            self.line.draw(img, color=color, thickness=3)
        else:
            self.line.draw(img, color=color, thickness=1)

        # point_a = (int(self.line.midpoint[0]), int(self.line.midpoint[1]))
        # point_b = (int(self.closest_point[0]), int(self.closest_point[1]))
        
        # cv2.line(img, point_a, point_b, color, 1)
        # cv2.line(img, (int(self.shape_line.point_a[0]), int(self.shape_line.point_a[1])),(int(self.shape_line.point_b[0]), int(self.shape_line.point_b[1])), color, 1)

class BarrierTermination():
    def __init__(self, source, destination, intersection, from_a, distance, is_virtual=False):
        self.source = source
        self.destination = destination
        self.intersection = int(intersection[0]), int(intersection[1])
        self.from_a = from_a
        self.distance = distance
        self.is_virtual = is_virtual

    @property
    def origin(self):
        return self.source.bounds.line.point_a if self.from_a else self.source.bounds.line.point_b

    def debug(self, img, color):
        cv2.line(img, self.origin, self.intersection, color, 1)

b_index = 0

class BarrierGroup():
    def __init__(self, barrier):
        self.barriers = [barrier]
        self.dead = False
        self._points = None
        self._bounds = None
        self._surfaces = None
        self.origin_barrier = barrier
        self.origin_surface = barrier.surface_barrier.surface
        self.vanishing_point = barrier.vanishing_point

        self.a_terminations = []
        self.a_termination_candidates = []
        self.term_a = None

        self.b_terminations = []
        self.b_termination_candidates = []
        self.term_b = None

        self._linkage = None

        global b_index
        self.index = b_index
        b_index += 1

    def add_barrier(self, barrier):
        self.barriers.append(barrier)

    def add_termination_a(self, termination:BarrierTermination):
        if termination not in self.a_terminations:
            self.a_terminations.append(termination)

    def add_termination_b(self, termination:BarrierTermination):
        if termination not in self.b_terminations:
            self.b_terminations.append(termination)

    @property
    def linkage(self):
        if self._linkage is None:
            self._linkage = set()
            for term in self.terminations:
                self._linkage.add(term.source)
                self._linkage.add(term.destination)

        return self._linkage

    @property
    def terminations(self):
        return self.a_terminations + self.b_terminations

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

    @property
    def points(self):
        if self._points is None:
            points = []
            for barrier in self.barriers:
                points.append(barrier.line.point_a)
                points.append(barrier.line.point_b)
            self._points = np.array(points)
        return self._points

    @property
    def bounds(self) -> RotatedRect:
        if self._bounds is None:
            self._bounds = RotatedRect(cv2.minAreaRect(self.points))
        return self._bounds

    @property
    def surfaces(self):
        if self._surfaces is None:
            self._surfaces = [self.origin_surface]

            for barrier in self.barriers:
                if barrier.surface_neighbor is not None and barrier.surface_neighbor not in self._surfaces:
                    self._surfaces.append(barrier.surface_neighbor)

            # for termination in self.terminations:
            #     surface = termination.barrier_group.origin_surface
            #     if not surface.surfaceType.is_major and surface not in self._surfaces:
            #         self._surfaces.append(surface)
        
        return self._surfaces

    def like(self, other, angle_threshold=np.radians(7), width_offset=0.0):
        
        angle = LineFunctions.line_angle_difference(self.bounds.line.angle, other.bounds.line.angle)

        if angle > angle_threshold:
            return False

        if self.surface_neighbor != other.surface_neighbor:
            return False

        # if compare_neighbors:
        #     if self.surface_neighbor != other.surface_neighbor:
        #         return False
        # else:
        #     for surface in self.surfaces:
        #         if surface not in other.surfaces:
        #             return False

        rect_a = self.bounds.resized(width_offset=width_offset)
        rect_b = other.bounds.resized(width_offset=width_offset)

        intersection, _ = cv2.rotatedRectangleIntersection(rect_a, rect_b)
        
        return intersection != 0

    def merge(self, other):
        self.barriers = list(set(self.barriers) | set(other.barriers))
        self._points = None
        self._bounds = None #trigger recalculation
        self.a_terminations = []
        self.b_terminations = []

    def debug(self, img, color, show_bounds=True):

        for barrier in self.barriers:
            barrier.debug(img, color=color)

        thickness = min(self.bounds.width, self.bounds.height)

        if show_bounds and thickness > 3:
            cv2.drawContours(img, [self.bounds.points], 0, (255,0,0), 1)

        self.bounds.line.draw(img, color=color)

        if not self.term_a is None:
            self.term_a.debug(img, color)

        if not self.term_b is None:
            self.term_b.debug(img, color)

    def debug_intersections(self, img):
        intersections = list(map(lambda x:x.intersection, self.terminations))

        thickness = min(self.bounds.width, self.bounds.height)
        radius = 5

        #connected points are small and blue
        for intersection in intersections:
            cv2.circle(img, intersection, radius, [0, 0, 255])  

        #open endpoints are larger and green
        #while closed are smaller and red
        if len(self.a_terminations) == 0: #open
            cv2.circle(img, self.bounds.line.point_a, int(radius * 2.0), [0, 255, 0]) 
        else: #closed
            cv2.circle(img, self.bounds.line.point_a, int(radius * 1.5), [255, 0, 0]) 

        if len(self.b_terminations) == 0: #open
            cv2.circle(img, self.bounds.line.point_b, int(radius * 2.0), [0, 255, 0])  
        else: #closed
            cv2.circle(img, self.bounds.line.point_b, int(radius * 1.5), [255, 0, 0]) 

        if debug_show_indices:
            put_text(img, str(self.index), self.bounds.line.midpoint, (0,0, 255), size=0.5, shadow=True)

def inside_mask(mask, point):
    if point[0] < mask.shape[1] and point[1] < mask.shape[0]:
        return mask[int(point[1]), int(point[0])] > 0
    return False

def get_distances(rect_a, rect_b, point):
    values = []

    values.append(distance.euclidean(point, rect_a.line.point_a))
    values.append(distance.euclidean(point, rect_a.line.point_b))
    values.append(distance.euclidean(point, rect_b.line.point_a))
    values.append(distance.euclidean(point, rect_b.line.point_b))

    return np.array(values)

#order matters for mean, grab two closest to bounds_a midpoint
def get_bounds_intersection(bounds_a, bounds_b):
    result, vertices = cv2.rotatedRectangleIntersection(bounds_a, bounds_b)

    if vertices is None:
        return None

    #average of the closest two:
    #get_loc = np.mean(vertices, axis=(0,1))
    vertices = list(vertices)
    vertices.sort(key=lambda x:distance.sqeuclidean(bounds_a.line.midpoint, x))
    vertices = np.array(vertices)

    return np.mean(vertices[:2], axis=(0,1))

class SurfaceBarriers():
    def __init__(self, data, surface, vanishing_points):
        self.data = data
        self.surface = surface
        self.surface.barriers = self
        self.vanishing_points = vanishing_points
        self.image = self.data["downscaled"]
        self.room = self.data["room"]
        self.diagonal = math.hypot(self.image.shape[0], self.image.shape[1])
        
        self.barrier_candidates = self.get_barrier_candidates()
        self.barrier_groups = self.group_barriers()
        self.merge_barriers()

    def getBarrier(self, index, elements=None):
        return next(filter(lambda barrier: barrier.index == index, self.barrier_groups if elements is None else elements), None)
        
    def refine(self, all_barriers):
        global debug_objects

        max_angle_parallel = np.radians(13)
        max_angle_orth = np.radians(30)
        min_distance = self.diagonal / 200
        min_distance_sq = min_distance * min_distance
        epsilon = min_distance

        self.set_initial_endpoints()
        self.cull_barriers(all_barriers)

        #now extend and link all:
        def get_best_termination(terminations):            
            if len(terminations) == 0:
                return None

            #get closest distance
            terminations.sort(key=lambda x: x.distance)
            best_match = terminations[0]

            #print("match")

            return best_match

        def is_valid_terimation(term, debug=False, secondary_match_threshold=0.5):
            if term is None:
                return False

            #(dist_a, intersection, barrier_b, is_colinear, is_virtual)
            num_points = int(max(term.distance // 5, 7))


            #check labels:
            samples = LineFunctions.get_line_samples(term.origin, term.intersection, self.room.semantic_labels, num_points)
            if len(samples) == 0:
                return False

            vals, counts = np.unique(samples, return_counts=True)

            values_counts = zip(vals.tolist(), counts.tolist())
            values_counts = sorted(values_counts, key=lambda x:x[1], reverse=True) #highest first

            semantic_matches = values_counts[0][0] + ADE20K.value_offset() == self.surface.bestLabel.value or \
                (len(values_counts) > 1 and (values_counts[1][0] + ADE20K.value_offset()) == self.surface.bestLabel.value and values_counts[1][1] / values_counts[0][1] >= secondary_match_threshold)

            if not semantic_matches:
                return False

            #check index mask, aka not too much of another surface:
            samples = LineFunctions.get_line_samples(term.origin, term.intersection, self.room.index_mask, num_points)
            if len(samples) == 0:
                return False

            vals, counts = np.unique(samples, return_counts=True)

            values_counts = zip(vals.tolist(), counts.tolist())
            values_counts = sorted(values_counts, key=lambda x:x[1], reverse=True) #highest first

            surface_matches = values_counts[0][0] == self.surface.index or \
                (len(values_counts) > 1 and values_counts[1][0] == self.surface.index and values_counts[1][1] / values_counts[0][1] >= secondary_match_threshold)

            return surface_matches

        # candidates = self.barrier_groups.copy()

        # for neighbor in self.surface.neighbors:
        #     if neighbor.barriers is not None:
        #         candidates.extend(filter(lambda x: x.vanishing_point in self.surface.vanishing_points, neighbor.barriers.barrier_groups))

        for barrier_a in self.barrier_groups:

            a_open = len(barrier_a.a_terminations) == 0
            b_open = len(barrier_a.b_terminations) == 0

            if not a_open and not b_open: continue

            bounds_a = RotatedRect(barrier_a.bounds.line.extended(1.05).bounding_box(min_distance))

            line_a = barrier_a.bounds.line.extended(3.0, from_a=a_open, from_b=b_open)
            rect_a = RotatedRect(line_a.bounding_box(min_distance))

            #find closest, either orthagonal or colinear and at the end, that's the one used, others ignored, 
            #This is across all elements

            for barrier_b in self.barrier_groups:
                if barrier_a == barrier_b or barrier_a.bounds.line.equals(barrier_b.bounds.line, epsilon): continue

                bounds_b = RotatedRect(barrier_b.bounds.line.extended(1.1).bounding_box(min_distance))

                if bounds_a.intersects(bounds_b): continue

                rect_b = RotatedRect(barrier_b.bounds.line.bounding_box(min_distance))

                intersection = get_bounds_intersection(rect_a, rect_b)

                is_virtual = False

                if intersection is None:
                    rect_b = RotatedRect(barrier_b.bounds.line.extended(1.5, from_a=len(barrier_b.a_terminations)==0, from_b=len(barrier_b.a_terminations)==0).bounding_box(min_distance))
                    intersection = get_bounds_intersection(rect_a, rect_b)
                    is_virtual = True


                if intersection is None: continue

                dist_a = distance.euclidean(intersection, barrier_a.bounds.line.point_a)
                dist_b = distance.euclidean(intersection, barrier_a.bounds.line.point_b)

                if dist_a < dist_b:
                    if a_open:
                        barrier_a.a_termination_candidates.append(BarrierTermination(barrier_a, barrier_b, intersection=intersection, from_a=True, distance=dist_a, is_virtual=is_virtual))
                else:
                    if b_open:
                        barrier_a.b_termination_candidates.append(BarrierTermination(barrier_a, barrier_b, intersection=intersection, from_a=False, distance=dist_b, is_virtual=is_virtual))

                #barrier_b counterpart
                a_open = len(barrier_b.a_terminations) == 0
                b_open = len(barrier_b.b_terminations) == 0

                dist_a = distance.euclidean(intersection, barrier_b.bounds.line.point_a)
                dist_b = distance.euclidean(intersection, barrier_b.bounds.line.point_b)

                if dist_a < dist_b:
                    if a_open:
                        barrier_b.a_termination_candidates.append(BarrierTermination(barrier_b, barrier_a, intersection=intersection, from_a=True, distance=dist_a, is_virtual=True))
                else:
                    if b_open:
                        barrier_b.b_termination_candidates.append(BarrierTermination(barrier_b, barrier_a, intersection=intersection, from_a=False, distance=dist_b, is_virtual=True))


        #get best
        for barrier in self.barrier_groups:

            term_a = get_best_termination(barrier.a_termination_candidates)

            if is_valid_terimation(term_a):
                if term_a.distance < min_distance * 2:
                    barrier.add_termination_a(term_a)
                else:
                    barrier.term_a = term_a

            term_b = get_best_termination(barrier.b_termination_candidates)

            if is_valid_terimation(term_b):
                if term_b.distance < min_distance * 2:
                    barrier.add_termination_b(term_b)
                else:
                    barrier.term_b = term_b

        
        self.barrier_groups = list(filter(lambda x: not x.dead, self.barrier_groups))


    def get_barrier_candidates(self, angle_threshold=np.radians(45)):

        padding = 10
        surface_mask = cv2.copyMakeBorder(self.surface.mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0)
        trans = cv2.distanceTransform(surface_mask, cv2.DIST_L2, 5)
        _, shape_mask = cv2.threshold(trans, 0.1 * trans.max(), 1, 0)

        shape_mask = shape_mask[padding:-padding,padding:-padding].astype(np.uint8)
        contours, _ = cv2.findContours(shape_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        self.shapes = list(map(lambda contour: cv2.approxPolyDP(contour, 0.003 * cv2.arcLength(contour, True), True), contours))
        self.candidates = []

        for line in self.data["lines"]:

            if line_on_image_edge(line.point_a, line.point_b, self.image):
                continue

            point_a = (line.point_a[0] + line.midpoint[0]) / 2, (line.point_a[1] + line.midpoint[1]) / 2
            point_b = (line.point_b[0] + line.midpoint[0]) / 2, (line.point_b[1] + line.midpoint[1]) / 2

            if inside_mask(self.surface.mask_edges, line.midpoint) and (inside_mask(self.surface.mask_edges, point_a) or inside_mask(self.surface.mask_edges, point_b)):
                self.candidates.append(line)
            # elif self.surface.parent is not None:
            #     if self.surface.parent.surfaceType == SurfaceType.Wall:
            #         self.candidates.append(line)

                #self.candidates.append(line)
                #print('%s: Search within parent %s' % (self.surface.name, self.surface.parent.name))

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

        search_width=self.diagonal/100
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

    def filter_barriers(self):
        barrier_groups, dead_barrier_groups = partition(lambda x: not x.dead, self.barrier_groups)
        for x in dead_barrier_groups: 
            x.dead = False #reset
        self.barrier_groups = barrier_groups

    def merge_barriers(self):

        width_offset = self.diagonal / 80
        #angle_threshold = np.radians(13)

        #merge similar barriers into one
        for i in range(len(self.barrier_groups)):
            group_a = self.barrier_groups[i]

            if group_a.dead: continue

            rect_a = RotatedRect(group_a.bounds.line.bounding_box(width_offset))
            angle_threshold_a = np.arctan(width_offset / group_a.bounds.line.length)
            
            for j in range(i + 1, len(self.barrier_groups)):
                group_b = self.barrier_groups[j]

                angle_threshold_b = np.arctan(width_offset / group_b.bounds.line.length)

                angle_threshold = max(max(angle_threshold_a, angle_threshold_b), np.radians(13))

                if line_angle_difference(group_a.bounds.line.angle, group_b.bounds.line.angle) > angle_threshold:
                    continue

                rect_b = RotatedRect(group_b.bounds.line.bounding_box(width_offset))
                
                if rect_a.intersects(rect_b):
                    group_a.merge(group_b)
                    group_b.dead = True

        self.filter_barriers()

    def set_initial_endpoints(self):
        #find interlinking
        min_distance = self.diagonal / 200
        min_length = self.diagonal / 20
        max_angle_parallel = np.radians(15)
        epsilon = min_distance

        #start from barrier groups, but also add sibling barriers to end
        candidates = self.barrier_groups.copy()
        # for neighbor in self.surface.neighbors:
        #     if neighbor.barriers is not None:
        #         candidates.extend(filter(lambda x: x.vanishing_point in self.surface.vanishing_points, neighbor.barriers.barrier_groups))

        #set termination points
        for i in range(len(self.barrier_groups)):
            barrier_a = self.barrier_groups[i]

            if barrier_a.dead or barrier_a.bounds.line.length < min_length: continue

            rect_a = RotatedRect(barrier_a.bounds.line.bounding_box(min_distance))

            a_terms = []
            b_terms = []
            merged = False

            for j in range(len(candidates)):
                barrier_b = candidates[j]

                if barrier_a == barrier_b or barrier_b.dead: continue
                #if barrier_b.vanishing_point not in self.surface.vanishing_points: continue

                is_colinear = LineFunctions.line_angle_difference(barrier_a.bounds.line.angle, barrier_b.bounds.line.angle) <= max_angle_parallel

                if is_colinear:
                    rect_b = RotatedRect(barrier_b.bounds.line.extended(1.2).bounding_box(min_distance))
                else:
                    rect_b = RotatedRect(barrier_b.bounds.line.bounding_box(min_distance))

                intersection = get_bounds_intersection(rect_a, rect_b)

                if intersection is None: 
                    continue

                dist_a = distance.euclidean(intersection, barrier_a.bounds.line.point_a)
                dist_b = distance.euclidean(intersection, barrier_a.bounds.line.point_b)
                    
                if is_colinear:
                    barrier_a.merge(barrier_b)
                    barrier_b.dead = True
                    merged = True
                    break
                else:
                    if dist_a < dist_b:
                        barrier_a.add_termination_a(BarrierTermination(barrier_a, barrier_b, intersection, from_a=True, distance=dist_a))
                    else:
                        barrier_a.add_termination_b(BarrierTermination(barrier_a, barrier_b, intersection, from_a=False, distance=dist_b))

            if merged: i -= 1 #recheck

        self.barrier_groups = list(filter(lambda x: not x.dead, self.barrier_groups))


    def cull_barriers(self, all_barriers):

        contours, contour_lengths = self.room.contours[self.surface.surfaceType]

        elementA = self.getBarrier(242)
        padding = self.diagonal / 100

        def closest_contour():
            #numerator/denominator distance to either point_a and point_b instead of those points, e.g. 3/4ths of the way to point_a instead of using point_a
            numerator = 7
            denominator = numerator + 1

            closest_inside = 100000
            closest_inside_index = None
            closest_outside = 100000
            closest_outside_index = None

            point_a = (numerator * group.bounds.line.point_a[0] + group.bounds.line.midpoint[0]) // denominator, (numerator * group.bounds.line.point_a[1] + group.bounds.line.midpoint[1]) // denominator
            point_b = (numerator * group.bounds.line.point_b[0] + group.bounds.line.midpoint[0]) // denominator, (numerator * group.bounds.line.point_b[1] + group.bounds.line.midpoint[1]) // denominator

            inside_padding = 5 + min(group.bounds.width, group.bounds.height) / 2

            for i in range(len(contours)):
                contour = contours[i]

                #positive (inside), negative (outside), or zero (on an edge)
                dist_midpoint = cv2.pointPolygonTest(contour, group.bounds.line.midpoint, True)
                dist_a = cv2.pointPolygonTest(contour, point_a, True)
                dist_b = cv2.pointPolygonTest(contour, point_b, True)

                #get absolute max between midpoint, point_a, and point_b
                dist = min(abs(dist_midpoint), min(abs(dist_a), abs(dist_b)))
                is_outside = dist_midpoint < -inside_padding

                if is_outside: #outside
                    if dist < closest_outside:
                        closest_outside = dist
                        closest_outside_index = i
                else:
                    if dist < closest_inside:
                        closest_inside = dist
                        closest_inside_index = i

            if closest_outside < closest_inside: #aka if group is outside
                #not inside, distance outside, contour index
                return False, closest_outside, closest_outside_index
            else:
                #is inside, distance outside, contour index
                return True, closest_inside, closest_inside_index
            

        good = [] #inside and within good distance given thresholds above
        bad = [] #inside and beyond distance deemed "good" by being under threshold
        ugly = [] #outside and too far away

        outside_threshold = 0.05
        inside_threshold = 0.1

        for i in range(len(self.barrier_groups)):
            group = self.barrier_groups[i]

            for j in range(i+1, len(self.barrier_groups)):
                group_b = self.barrier_groups[j]
                if group.bounds.line.equals(group_b.bounds.line, 3):
                    group.dead = True
                    break

            if group.dead: continue
            is_inside, dist, index = closest_contour()
            if dist < padding:
                is_inside = True

            # if group.index == 242:
            #     print(is_inside, dist)

            if is_inside: #aka if group is inside
                #if inside but beyond threshold distance
                if dist > max(inside_threshold * contour_lengths[index], 3):
                    ugly.append(group)
                else:
                    good.append(group)
            else:
                #if outside and beyond threshold distance
                if dist > max(outside_threshold * contour_lengths[index], 3):
                    ugly.append(group)
                else:
                    bad.append(group)

        if len(good) > 0:
            seeds = good
        elif len(bad) > 0:
            seeds = bad
        else: #do nothing, do not trust culling result
            return

        self.barrier_groups = list(filter(lambda x: not x.dead, self.barrier_groups))

        valid = self.barrier_groups.copy()

        #flood fill from seeds set into valid set using barrier linkage
        seeds = set(seeds)
        checked = seeds.copy()
        keep = seeds.copy()

        def matches(element, candidate):
            if candidate in checked or candidate.vanishing_point not in self.surface.vanishing_points:
                return False

            #links back into the series of valid elements, but not just back to element
            links_back = next(filter(lambda barrier_group: barrier_group in valid and barrier_group != element, candidate.linkage), None)

            if links_back is None: return False

            return True

        while len(seeds) > 0:
            element = seeds.pop()

            for candidate in element.linkage:
                if matches(element, candidate):
                    if candidate in valid:
                        seeds.add(candidate)
                    keep.add(candidate)

                checked.add(candidate)

        
        self.barrier_groups = list(keep)
        elementB = self.getBarrier(242)

        if elementA is not None and elementB is None:
            print("Deleted element", elementA.index, elementA.bounds.line.point_a, elementA.bounds.line.point_b)


    def debug(self, img, color):
        for shape in self.shapes:
            cv2.drawContours(img, [shape], -1, color=(255,255,255), thickness=1)

        #Line.draw_all(img, self.candidates, color=(0,0,0), thickness=1)

        # for barrier in self.barrier_candidates:
        #     barrier.debug(img, color=color)

        for barrier_group in self.barrier_groups:
            barrier_group.debug(img, color=color)

class JunctionType(IntEnum):
    Extension = 0
    Vertex = 1
 
class BarrierJunction():
    def __init__(self, data, barrier_groups:list, junction_type:JunctionType):
        super().__init__()
        self.barrier_groups = barrier_groups
        self.junction_type = junction_type

class BarrierSolver():
    def __init__(self, data, surface_barriers):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface_barriers = surface_barriers
        self.diagonal = math.hypot(self.image.shape[0], self.image.shape[1])

    def solve(self):
        
        def get_surface_barriers(surface):
            return list(filter(lambda group: surface in group.surfaces, self.barrier_groups))

        def log_barriers(surfaces, name):

            if len(surfaces) == 0:
                return

            debug = self.image.copy()

            for surface in surfaces:
                color = random_color()
                
                for group in surface.barriers.barrier_groups:
                    group.debug(debug, color=color, show_bounds=True)

            for surface in surfaces:

                for group in surface.barriers.barrier_groups:
                    group.debug_intersections(debug)

            for obj in debug_objects:
                if obj is None: continue

                color =  random_color()
                if isinstance(obj, RotatedRect):
                    cv2.drawContours(debug, [obj.points], 0, color, 2)
                elif isinstance(obj, Line):
                    #print(obj.point_a, obj.point_b)
                    obj.draw(debug, color)
                elif isinstance(obj, BarrierTermination):
                    obj.debug(debug, color)
                elif len(obj) == 2:
                    cv2.circle(debug, (int(obj[0]), int(obj[1])), 10, color) 
                else:
                    print("Skipping", obj)
                
            log_image(self.data, name + "_barriers", debug)

        #flatten groups
        self.barrier_groups = []
        for sb in self.surface_barriers.values():
            self.barrier_groups.extend(sb.barrier_groups)

        #remove redundant                                       

        #extend to other lines that are not terminated to other colinear lines
        
        max_angle_parallel = np.radians(15)
        max_angle_orth = np.radians(30)

        for i in range(len(self.barrier_groups)):
            barrier_a = self.barrier_groups[i]

            aa_may_extend = len(barrier_a.a_terminations) == 0
            ab_may_extend = len(barrier_a.b_terminations) == 0

            a_may_extend = aa_may_extend or ab_may_extend

            rect_a = None

            continue

            for j in range(i+1, len(self.barrier_groups)):
                barrier_b = self.barrier_groups[j]

                if barrier_a.origin_surface.bestLabel != barrier_b.origin_surface.bestLabel: continue

                ba_may_extend = len(barrier_b.a_terminations) == 0
                bb_may_extend = len(barrier_b.b_terminations) == 0
                b_may_extend = ba_may_extend or bb_may_extend

                if not a_may_extend and not b_may_extend: continue

                colinear = LineFunctions.line_angle_difference(barrier_a.bounds.line.angle, barrier_b.bounds.line.angle) <= max_angle_parallel
                orthagonal = LineFunctions.line_angle_difference(barrier_a.bounds.line.angle, barrier_b.bounds.line.angle + 0.5 * np.pi) <= max_angle_orth

                if not colinear or orthagonal: continue

                #increase line length:
                if rect_a is None:
                    rect_a = barrier_a.bounds.resized(width_offset=max_distance, width_factor=0, length_offset=max_distance, length_factor=3.0)
                rect_b = barrier_b.bounds.resized(width_offset=max_distance, width_factor=0, length_offset=max_distance, length_factor=3.0)

                intersection = rect_a.get_intersection(rect_b)
                if intersection is None: continue

                distances = get_distances(barrier_a.bounds, barrier_b.bounds, intersection)

                aa_intersection = distances[0] <= max_distance
                ab_intersection = distances[1] <= max_distance
                a_intersection = aa_intersection or ab_intersection

                ba_intersection = distances[2] <= max_distance
                bb_intersection = distances[3] <= max_distance
                b_intersection = ba_intersection or bb_intersection

               
        if im_logging_enabled(self.data):
            
            log_barriers(self.room.get_surfaces(surfaceTypes=[SurfaceType.WallLike]), "wall_like")

            log_barriers(self.room.get_surfaces(surfaceTypes=[SurfaceType.Wall]), "wall") 

            # log_barriers(self.room.get_surfaces(surfaceTypes=[SurfaceType.Floor]), "floor")

            # log_barriers(self.room.get_surfaces(surfaceTypes=[SurfaceType.Ceiling]), "ceiling")         

            log_barriers(self.room.get_surfaces(labels=box_like), "box")
            
        
        
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
        self.surfaces.extend(self.room.get_surfaces(surfaceTypes=[SurfaceType.Wall, SurfaceType.WallLike, SurfaceType.Ceiling, SurfaceType.Floor]))
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

        all_barriers = []
        for surface in self.surfaces:
            all_barriers.extend(surface.barriers.barrier_groups)

        for surface in self.surfaces:
            self.barriers[surface.uniqueId].refine(all_barriers)

        # bs = BarrierSolver(self.data, self.barriers)
        # bs.solve()

        self.refine_masks()


    def refine_masks(self):

        print("refining")

        watershed_image = self.image.copy()

        markers = np.zeros(self.image.shape, dtype=np.int32)
        watershed_mask = np.zeros(self.image.shape, dtype=np.int32)

        for surface in self.surfaces:
            mask = surface.mask
            markers[mask > 0] = index + 1

        markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))



    def get_debug_image(self):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.room.surfaces))

        #overlay probs

        for i in range(len(self.room.surfaces)):
            surface = self.room.surfaces[i]
            
            #mask = (self.barriers[surface.uniqueId].mask_edges if surface.uniqueId in self.barriers else surface.mask) > 0
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
