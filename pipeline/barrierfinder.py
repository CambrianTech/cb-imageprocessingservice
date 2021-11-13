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
from .Line import line_angle_difference, Line, on_image_edge
from .room import Room, Surface

class BarrierLine(Line):
    def __init__(self, data, group, start_index, stop_index):
        super().__init__(data, group=group)
        self.start_index = start_index
        self.stop_index = stop_index

class Barrier():
    def __init__(self, line, poly_length):
        self.line = line
        self.poly_length = poly_length
        self.source_lines = [line]
        self.indices = [line.id]
        self.destroyed = False

    def intersects(self, line, search_width=13, length_multiplier=5.0):
        if line.group != self.line.group:
            return False

        rect_a = self.line.bounding_box(search_width, length_multiplier=length_multiplier)
        rect_b = line.bounding_box(search_width, length_multiplier=length_multiplier)
        result, _ = cv2.rotatedRectangleIntersection(rect_a, rect_b)

        return result != 0

    @property
    def start_index(self):
        return self.line.start_index % self.poly_length

    @property
    def stop_index(self):
        return self.line.stop_index % self.poly_length

    @property
    def length(self):
        return self.line.stop_index - self.line.start_index

    def merge(self, line:BarrierLine, merged_data=None):
        if merged_data is None:
            merged_data = LineFunctions.merge_lines((self.line.point_a, self.line.point_b), (line.point_a, line.point_b))

        start_index = min(self.line.start_index, line.start_index)
        stop_index = max(self.line.stop_index, line.stop_index)
        self.line = BarrierLine(np.array([merged_data[0][0], merged_data[0][1], merged_data[1][0], merged_data[1][1]], dtype=np.int), group=line.group, start_index=start_index, stop_index=stop_index)
        self.source_lines.append(line)

    def extend_to(self, line):
        print("Extend to point")


class BarrierFinder():
    def __init__(self, data, surface):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface = surface

    def line_semantics(self, point_a, point_b):
        merged_result_len = distance.euclidean(point_a, point_b)
        num_samples = int(merged_result_len + 1)
        samples = LineFunctions.get_line_samples(point_a, point_b, self.surface.geometry.semantic_labels, num_samples)
        
        unique, counts = np.unique(samples, return_counts=True)
        unique_counts = zip(unique.tolist(), counts.tolist())
        return sorted(unique_counts, key=lambda x:x[1], reverse=True)

    def find_merged_match(self, line, search_width, barriers, angle_threshold):

        match = next(filter(lambda x: x.intersects(line, search_width=search_width), barriers), None)

        if match is None: return None, None

        if LineFunctions.line_angle_difference(line.angle, match.line.angle) > angle_threshold:
            return None, None

        merged_result = LineFunctions.merge_lines((match.line.point_a, match.line.point_b), (line.point_a, line.point_b))
        merged_semantics = self.line_semantics(merged_result[0], merged_result[1])
        merged_label = ADE20K(merged_semantics[0][0] + 1)
        merged_counts_ratio = merged_semantics[0][1] / (merged_semantics[0][1] + merged_semantics[1][1]) if len(merged_semantics) > 1 else 1

        #print(self.surface.name, merged_label, merged_counts_ratio)
        
        if self.surface.bestLabel == merged_label and merged_counts_ratio > 0.7:
            return match, merged_result

        return None, None


    def border_search(self, poly, clockwise, vps, min_length, max_length, angle_threshold):

        min_length_sq = min_length * min_length
        max_length_sq = max_length * max_length
        theta_thresh = np.cos(angle_threshold)

        last_vp_match = None

        num_pts = len(poly)
        group = 0

        barriers = []

        diagonal = math.hypot(self.image.shape[0], self.image.shape[1])

        search_width = int(diagonal / 50)

        for i in range(num_pts):

            index_a = i if clockwise else (num_pts - i - 1)
            index_b = (i+1 if clockwise else (num_pts - i - 2)) % num_pts

            point_a = poly[index_a][0]
            point_b = poly[index_b][0]

            length_sq = distance.sqeuclidean(point_a, point_b)

            if length_sq < min_length_sq or length_sq > max_length_sq:
                continue

            if on_image_edge(point_a, self.image) and on_image_edge(point_a, self.image) & on_image_edge(point_b, self.image) > 0:
                continue
            
            line = BarrierLine(np.array([point_a[0], point_a[1], point_b[0], point_b[1]]), group=group, start_index=i, stop_index=i+1)

            directions = np.array([line.direction]) 
            directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
            locations = np.array([line.midpoint])

            vp_match = next(filter(lambda vp: angle_with_vp(vp.model, locations, directions) > theta_thresh, vps), None)

            if vp_match is None: continue

            est_directions = locations - vp_match.direction

            direction = normalize(est_directions[0]) * 0.5 * line.length
            est_line = BarrierLine(np.array([line.midpoint[0] - direction[0], line.midpoint[1] - direction[1], line.midpoint[0] + direction[0], line.midpoint[1] + direction[1]], dtype=np.int), \
                group=group, start_index=i, stop_index=i+1) #i+1 may extend into start by modulous division, but must be kept track of

            if LineFunctions.line_angle_difference(line.angle, est_line.angle) <= angle_threshold / 2:
                line = est_line

            if last_vp_match != None and last_vp_match != vp_match:
                #made a legit turn, do not allow grouping with past barriers (could check angle?)
                #delete ones we've skipped over here
                group += 1

            last_vp_match = vp_match

            match, merged_result = self.find_merged_match(line, search_width, barriers, angle_threshold)

            if match is not None:
                match.merge(line, merged_result)
            else:
                barriers.append(Barrier(line, poly_length=num_pts))

        return barriers

    def reintegrate_barriers(self, poly, barriers, occupied=None):

        if occupied is None:
            occupied = np.full(len(poly), False)

        filtered = []

        for barrier in sorted(barriers, key=lambda x:x.length, reverse=True):

            if barrier.start_index < barrier.stop_index:
                count = np.count_nonzero(occupied[barrier.start_index:barrier.stop_index])
            else:
                count = np.count_nonzero(occupied[barrier.start_index:]) + np.count_nonzero(occupied[0:barrier.stop_index])

            if count > 0:
                #print({"count":count, "start":barrier.start_index, "stop":barrier.stop_index}, {"barrier len": barrier.length, "poly len": len(poly)})
                continue

            if barrier.start_index < barrier.stop_index:
                occupied[barrier.start_index:barrier.stop_index] = True
            else:
                occupied[barrier.start_index:] = True
                occupied[0:barrier.stop_index] = True            

            filtered.append(barrier)


        return filtered, occupied
            #point_a = poly[i][0]
            #point_b = poly[(i+1) % num_pts][0]

    def inlier_search(self, poly, vp_lines, inner_padding, outer_padding):

        def on_the_border(line):
            #positive (inside), negative (outside), or zero (on an edge)
            dist_a = cv2.pointPolygonTest(poly, line.point_a, True)
            a_inside = dist_a > 0
            a_ok = dist_a < inner_padding if a_inside else abs(dist_a) < outer_padding

            dist_b = cv2.pointPolygonTest(poly, line.point_b, True)
            b_inside = dist_b > 0
            b_ok = dist_b < inner_padding if b_inside else abs(dist_b) < outer_padding

            dist_midpoint = cv2.pointPolygonTest(poly, line.midpoint, True)
            mid_inside = dist_midpoint > 0
            mid_ok = dist_midpoint < inner_padding if mid_inside else abs(dist_midpoint) < outer_padding            

            return mid_ok and (abs(dist_a - dist_midpoint) < inner_padding / 3 or abs(dist_b - dist_midpoint) < inner_padding / 3)

        inlier_lines = (list(filter(lambda line: on_the_border(line), vp_lines)))

        return inlier_lines
            
    def find(self, vp_lines, angle_threshold=np.radians(5), min_length=5, max_length=1000):
        
        if len(self.surface.vanishing_points) < 0:
            return []

        diagonal = math.hypot(self.image.shape[0], self.image.shape[1])

        #pull barriers from surface contours:

        #for vp in self.surface.vanishing_points:

        
        line_groups = []

        for poly in self.surface.polygons:
            #both directions:

            surface_barriers = []

            #clockwise:
            barriers = self.border_search(poly, True, self.surface.vanishing_points, min_length, max_length, angle_threshold)
            #remove ones skipped by others:
            barriers, occupied = self.reintegrate_barriers(poly, barriers)
            surface_barriers.extend(barriers)

            # #counter_clockwise:
            barriers = self.border_search(poly, False, self.surface.vanishing_points, min_length, max_length, angle_threshold)
            barriers, occupied = self.reintegrate_barriers(poly, barriers, occupied)
            surface_barriers.extend(barriers)

            #collect lines for polygon
            poly_lines = list(map(lambda x: x.line, surface_barriers))

            #pull barriers from inlier lines near surface edges:
            moments = cv2.moments(poly)
            area = moments['m00']

            centroid = (int(moments['m10'] / area), int(moments['m01'] / area)) if area > 0 else None

            length = math.sqrt(area) if area > 0 else diagonal / 50
            inner_padding = length / 10
            outer_padding = length / 10 
            poly_lines.extend(self.inlier_search(poly, vp_lines, inner_padding, outer_padding))
            
            poly_lines = Line.merge(poly_lines, search_width=diagonal/200, search_length=1.5, angle_threshold=math.radians(3))

            line_groups.append(poly_lines)
            
            # #filter out ones that have a match further away from the center but in the same direction
            # if centroid is None:
            #     continue

            # removed = []

            # for i in range(len(poly_lines)):
            #     line_a = poly_lines[i]
            #     matches = []

            #     for j in range(i+1, len(poly_lines)):
            #         line_b = poly_lines[j]

            #         if LineFunctions.line_angle_difference(line_a.angle, line_b.angle) > angle_threshold:
            #             continue

                    #now same angle:



        
        return line_groups

class BarrierSolver():
    def __init__(self, data, surface):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface = surface


    def solve(self, max_iterations=2000, threshold_inlier=math.radians(2), max_time=0.33, measure_area=False):     

        max_iterations = min(max_iterations, len(self.surface.barriers) * 40)
        start_time = time.time() 

        num_samples = random.randint(2, len(self.barriers))
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

        diagonal = math.hypot(self.image.shape[0], self.image.shape[1])

        self.surfaces = []
        self.surfaces.extend(self.room.get_surfaces(surfaceTypes=[SurfaceType.Wall, SurfaceType.WallLike]))
        self.surfaces.extend(self.room.get_surfaces(labels=box_like))

        self.lines = []

        def inside_mask(mask, point):
            if point[0] < mask.shape[1] and point[1] < mask.shape[0]:
                return expanded_mask[point[1], point[0]] > 0
            return False

        for surface in self.surfaces:
            surface.barrier_lines = []
            expanded_mask = adjust_mask(cv2.dilate, surface.mask)

            for line in self.data["lines"]:
                if inside_mask(expanded_mask, line.point_a):
                    surface.barrier_lines.append(line)

        # ceilings = self.room.get_surfaces(surfaceTypes=[SurfaceType.Ceiling])
        # for surface in ceilings:
        #     self.vp_lines.extend(surface.border_lines)

        #self.lines = Line.merge(self.lines, search_width=diagonal/200, search_length=1.2, angle_threshold=math.radians(5))

        if im_logging_enabled(self.data):
            log_image(self.data, "barriers.png", self.get_debug_image())


    def get_debug_image(self):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.room.surfaces))

        #overlay probs
        for i in range(len(self.room.surfaces)):
            surface = self.room.surfaces[i]
            mask = surface.mask > 0

            max_value = 0.9
            if max_value > 0:
                img_hsv[:, :, 0][mask] = hues[i]
                img_hsv[:, :, 1][mask] = 255 * np.power(surface.probs[mask], 0.15)
                    
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)

        for i in range(len(self.room.surfaces)):
            surface = self.room.surfaces[i]
            color = convert_color((hues[i],127,255), cv2.COLOR_HSV2RGB_FULL)
            Line.draw_all(img, surface.barrier_lines, color=color, thickness=2)

        return img
