import numpy as np
from scipy import ndimage
import cv2
import random
import time
from scipy.spatial import distance
import math

from .ade20k import ADE20K
from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .utils import resize_array, random_color, overlay_mask, normalize, convert_color, put_text
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
    def __init__(self, data, surface, hed):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface = surface
        self.hed = hed

    def border_search(self, poly, clockwise, vps, min_length, max_length, angle_threshold):

        min_length_sq = min_length * min_length
        max_length_sq = max_length * max_length
        theta_thresh = np.cos(angle_threshold)

        def _line_stats(point_a, point_b):
            merged_result_len = distance.euclidean(point_a, point_b)
            num_samples = int(merged_result_len + 1)
            samples = LineFunctions.get_line_samples(point_a, point_b, self.image, num_samples)
            return  np.mean(samples, axis=0), np.std(samples, axis=0)

        def line_stats(point_a, point_b):
            values = [_line_stats(point_a, point_b), \
                      _line_stats((point_a[0] - 1, point_a[1] - 1), (point_b[0] - 1, point_b[1] - 1)), \
                      _line_stats((point_a[0] + 1, point_a[1] + 1), (point_b[0] + 1, point_b[1] + 1))]

            values.sort(key=lambda x: np.linalg.norm(x[1]))
            return values[0]

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

            if on_image_edge(point_a, self.image) and on_image_edge(point_b, self.image):
                continue
            
            line = Line(np.array([point_a[0], point_a[1], point_b[0], point_b[1]]))

            directions = np.array([line.direction]) 
            directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
            locations = np.array([line.midpoint])

            vp_match = None

            for vp in vps:
                theta = angle_with_vp(vp.model, locations, directions)
                if theta > theta_thresh:
                    vp_match = vp
                    break

            if vp_match is None:
                continue

            est_directions = locations - vp_match.direction

            direction = normalize(est_directions[0]) * 0.5 * line.length
            line = BarrierLine(np.array([line.midpoint[0] - direction[0], line.midpoint[1] - direction[1], line.midpoint[0] + direction[0], line.midpoint[1] + direction[1]], dtype=np.int), \
                group=group, start_index=i, stop_index=i+1) #i+1 may extend into start by modulous division, but must be kept track of

            if last_vp_match != None and last_vp_match != vp_match:
                #made a legit turn, do not allow grouping with past barriers (could check angle?)
                #delete ones we've skipped over here
                group += 1

            last_vp_match = vp_match

            match = next(filter(lambda x: x.intersects(line, search_width=search_width), barriers), None)

            if match is not None:
                if LineFunctions.line_angle_difference(line.angle, match.line.angle) < angle_threshold:
                    #check for 
                    merged_result = LineFunctions.merge_lines((match.line.point_a, match.line.point_b), (line.point_a, line.point_b))

                    mean_line_a, std_line_a = line_stats(line.point_a, line.point_b)
                    mean_line_b, std_line_b = line_stats(match.line.point_a, match.line.point_b)
                    
                    diff = np.linalg.norm(mean_line_b - mean_line_a)

                    #mean = min(np.linalg.norm(mean_line_a), np.linalg.norm(mean_line_b))
                    std = min(np.linalg.norm(std_line_a), np.linalg.norm(std_line_b))

                    #cabinet 5: 42.04354087596048 38.03833631464182 245.3443258180998

                    threshold = std * 7
                    #print(self.surface.name, diff, threshold)

                    if diff < threshold:
                        match.merge(line, merged_result)
                        continue
            
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


    def solve(self, angle_threshold=np.radians(7), min_length=50, max_length=1000):
        
        if len(self.surface.vanishing_points) < 0:
            return []

        surface_barriers = []

        for poly in self.surface.polygons:
            #both directions:

            #clockwise:
            barriers = self.border_search(poly, True, self.surface.vanishing_points, min_length, max_length, angle_threshold)
            #remove ones skipped by others:
            barriers, occupied = self.reintegrate_barriers(poly, barriers)
            surface_barriers.extend(barriers)

            #counter_clockwise:
            barriers = self.border_search(poly, False, self.surface.vanishing_points, min_length, max_length, angle_threshold)
            barriers, occupied = self.reintegrate_barriers(poly, barriers, occupied)
            surface_barriers.extend(barriers)
        

        #link/merge barriers:
        diagonal = math.hypot(self.image.shape[0], self.image.shape[1])
        search_width = diagonal / 50

        for i in range(len(surface_barriers)):
            barrier_a = surface_barriers[i]

            best_match = None
            for j in range(i+1, len(surface_barriers)):
                barrier_b = surface_barriers[j]

                #barrier_b.intersects()


                
        return surface_barriers
        
        
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

        hed = cv2.resize(self.data["hed"], (self.image.shape[1], self.image.shape[0]))
        hed = cv2.normalize(hed, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        self.surfaces = []
        self.surfaces.extend(self.room.get_surfaces(surfaceTypes=[SurfaceType.Wall, SurfaceType.WallLike]))
        self.surfaces.extend(self.room.get_surfaces(labels=box_like))

        for surface in self.surfaces:
            bf = BarrierFinder(self.data, surface, hed)
            min_size = np.sqrt(surface.min_area) / 3
            max_size = np.sqrt(surface.max_area) * 2
            surface.barriers = bf.solve(min_length=min_size, max_length=max_size)

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

        theta_thresh = np.cos(np.radians(7))

        for i in range(len(self.room.surfaces)):
            surface = self.room.surfaces[i]
            color = convert_color((hues[i],127,255), cv2.COLOR_HSV2RGB_FULL)

            #cv2.drawContours(img, surface.polygons, -1, color, 1)

            for poly in surface.polygons:
                num_pts = len(poly)

                for j in range(num_pts):
                    point_a = poly[j][0]
                    point_b = poly[(j + 1) % num_pts][0]

                    line = Line(np.array([point_a[0], point_a[1], point_b[0], point_b[1]]))

                    directions = np.array([line.direction]) 
                    directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
                    locations = np.array([line.midpoint])

                    match = False

                    for vp in surface.vanishing_points:
                        theta = angle_with_vp(vp.model, locations, directions)
                        if theta > theta_thresh:
                            match = True
                            break
                    
                    if match:
                        cv2.line(img, point_a, point_b, color, thickness=1)

            
            #Line.draw_all(img, surface.horizontal_vp[0].inlier_lines, color=color, thickness=1)

            for barrier in surface.barriers:
                barrier.line.draw(img, color=color, thickness=3)

            for barrier in surface.barriers:
                cv2.drawMarker(img, barrier.line.point_a, color=color)
                cv2.drawMarker(img, barrier.line.point_b, color=color)

        return img


