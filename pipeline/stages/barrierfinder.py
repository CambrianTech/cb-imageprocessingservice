import numpy as np

import cv2
import random
import time
import math
from scipy.spatial import distance
from operator import attrgetter
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from skimage.morphology import remove_small_holes, remove_small_objects

from cambrian.LineFunctions import LineFunctions
from cambrian.image_processing import kmeans_image

from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.data.logging import log_image, im_logging_enabled, log_markers, get_segmentation_image, log_segmentation_image, log_mask
from pipeline.components.line import draw_lines
from .vanishingpointfinder import angle_with_vp
from pipeline.misc.utils import random_color, resize_array, adjust_mask, convert_color, put_text
from pipeline.data.ade20k import ADE20K, on_floor, on_wall, on_ceiling, box_like, legged_objects, lights
from pipeline.components.rotated_rect import RotatedRect
from pipeline.components.line import Line, extend_to_intersection, draw_lines, line_within_mask, line_on_image_edge, merge_lines
from pipeline.stages.vanishingpointfinder import get_inliers


class VerticalBarrierSet():
    def __init__(self, data, k, mask, wall_mask, invalid_mask):
        self.data = data
        self.k_means_constant = k

        self.mask = mask
        self.wall_mask = wall_mask
        self.invalid_mask = invalid_mask
        self.vp_angle_diff = np.radians(8)

    def find_polygons(self, labels, epsilon):

        pad = 10
        results = []
        mask_area = labels.shape[0] * labels.shape[1]

        self.labels = np.zeros_like(labels)

        total_length = 0

        for i in range(0, self.k_means_constant):
            mask = np.zeros_like(labels)
            mask[labels == i] = 1
            #mask[self.wall_mask == 0] = 0 

            area = cv2.countNonZero(mask)

            area_threshold = mask_area / 50

            mask = remove_small_holes(mask, area_threshold=area_threshold).astype(np.uint8)

            self.labels[mask > 0] = (i + 1)

            mask_bordered = cv2.copyMakeBorder(mask, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0) 
            contours, _ = cv2.findContours(image=mask_bordered, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE, offset=(-pad,-pad))

            total_length += sum(cv2.arcLength(contour, True) for contour in contours)

            test_mask = np.zeros_like(self.wall_mask)
            cv2.drawContours(test_mask, contours, -1, 1, thickness=cv2.FILLED)
            test_mask[self.wall_mask == 0] = 0

            results.append((cv2.countNonZero(test_mask), contours))

        results = sorted(results, key=lambda x: x[0], reverse=True)

        best_count, _ = results[0]

        polygons = []
        for result in results:
            if result[0] > best_count / 3:
                contours = result[1]

                for contour in contours:

                    test_mask = np.zeros_like(self.mask)
                    cv2.drawContours(test_mask, [contour], -1, 1, thickness=cv2.FILLED)

                    contour_area = cv2.countNonZero(test_mask)

                    invalid = cv2.bitwise_and(test_mask, self.invalid_mask)
                    invalid_area = cv2.countNonZero(invalid)

                    if invalid_area < contour_area / 2:
                        polygon = cv2.approxPolyDP(contour, epsilon, True)
                        polygons.append(polygon)

        return polygons, total_length

    def recluster(self, scale=1.0, epsilon=1.5):
        labels = self.data["surface_normals"]

        if scale != 1.0:
            _, self.normals_labels, _ = kmeans_image(cv2.resize(labels, (int(scale * labels.shape[1]), int(scale * labels.shape[0]))), self.k_means_constant)
            self.normals_labels = cv2.resize(self.normals_labels.astype(np.uint8), (labels.shape[1], labels.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            _, self.normals_labels, _ = kmeans_image(labels, self.k_means_constant)
            self.normals_labels = self.normals_labels.astype(np.uint8)

        self.polygons, self.contour_length = self.find_polygons(self.normals_labels, epsilon=epsilon)


    def calculate_score(self, scale=0.25, epsilon=3.0):
        
        self.recluster(scale, epsilon=epsilon)

        diagonal = math.hypot(self.mask.shape[0], self.mask.shape[1])
        area = self.mask.shape[0] * self.mask.shape[1]
        min_line_length = diagonal / 30

        good_lines = []
        bad_lines = []

        self.total_defects = 0

        for polygon in self.polygons:

            num_pts = len(polygon)

            try:
                hull = cv2.convexHull(polygon, returnPoints=False)
                defects = cv2.convexityDefects(polygon, hull)
                if defects is not None:
                    self.total_defects += defects.shape[0]
            except:
                pass

            for i in range(num_pts):
                point_a = polygon[i][0]
                point_b = polygon[(i+1) % num_pts][0]

                if line_on_image_edge(point_a, point_b, self.mask.shape[1], self.mask.shape[0], min_distance=15):
                    continue

                line = Line(point_a[0], point_a[1], point_b[0], point_b[1])

                length = distance.euclidean(point_a, point_b)

                if length >= 5:
                    good_lines.append(line)
                        
                bad_lines.append(line)

        vps = [self.data["vertical_vp"]]
        #vps.extend(self.data["horizontal_vps"])

        self.filtered_lines = []
        for vp in vps:
            self.filtered_lines.extend(get_inliers(good_lines, vp.model, self.vp_angle_diff))

        self.filtered_lines = merge_lines(self.filtered_lines, search_width=max(diagonal/150, 3), angle_threshold=self.vp_angle_diff)

        def line_valid(line):

            if line.length < min_line_length:
                return False

            test_length = int(line.length * 0.2)

            test_a = line.normal_a * test_length + line.midpoint
            test_b = line.normal_b * test_length + line.midpoint

            line_samples = LineFunctions.get_line_samples(line.point_a, line.point_b, self.wall_mask, max(3, int(length)))
            o_samples_a = LineFunctions.get_line_samples(line.midpoint, test_a, self.invalid_mask, max(3, test_length))
            o_samples_b = LineFunctions.get_line_samples(line.midpoint, test_b, self.invalid_mask, max(3, test_length))

            mean_line = np.mean(line_samples)
            o_mean_a = np.mean(o_samples_a)
            o_mean_b = np.mean(o_samples_b)

            return mean_line > 0.5 and o_mean_a < 0.5 and o_mean_b < 0.5

        #self.filtered_lines = list(filter(lambda line: line_valid(line), self.filtered_lines))

        self.total_line_length = sum(math.pow(line.length, 2) for line in self.filtered_lines)

        #cv2.countNonZero(ceiling_mask)

        print("defects k=%d, %d" % (self.k_means_constant, self.total_defects))
        print("good_lines k=%d" % self.k_means_constant, len(self.filtered_lines))


        self.bad_lines = list(get_inliers(bad_lines, self.data["vertical_vp"].model, self.vp_angle_diff))
        self.total_bad_line_length = sum(line.length for line in self.bad_lines)

        self.score = 0.0

        if self.contour_length > 0 and self.total_defects > 0:
            self.score = (self.k_means_constant * math.pow(self.total_line_length, 2.0)) / (self.contour_length * self.total_defects)
            
        #return 100.0 * (self.total_line_length - self.total_bad_line_length) / self.contour_length

        return self.score


    _vertical_lines = None

    @property
    def vertical_lines(self):
        if self._vertical_lines is None:
            good_lines = []

            diagonal = math.hypot(self.normals_labels.shape[0], self.normals_labels.shape[1])
            min_line_length = diagonal / 200

            for polygon in self.polygons:

                num_pts = len(polygon)

                for i in range(num_pts):
                    point_a = polygon[i][0]
                    point_b = polygon[(i+1) % num_pts][0]

                    if line_on_image_edge(point_a, point_b, self.normals_labels.shape[1], self.normals_labels.shape[0], min_distance=15):
                        continue

                    line = Line(point_a[0], point_a[1], point_b[0], point_b[1])

                    length = distance.euclidean(point_a, point_b)

                    if length >= min_line_length:
                        samples = LineFunctions.get_line_samples(point_a, point_b, self.mask, int(line.length / 3))
                        is_within_mask = np.mean(samples) > 0.5
                        
                        if is_within_mask:
                            good_lines.append(line)
                        
            self._vertical_lines = list(get_inliers(good_lines, self.data["vertical_vp"].model, self.vp_angle_diff))

            self._vertical_lines = merge_lines(self._vertical_lines, search_width=max(diagonal/150, 3), search_length=1.05, angle_threshold=self.vp_angle_diff)


        return self._vertical_lines

    def debug(self, data, name):

        debug = get_segmentation_image(self.labels, data["downscaled"], labelset=None)

        opacity = 0.25
        debug = cv2.addWeighted(debug, opacity, data["downscaled"], 1.0 - opacity, 0)

        cv2.drawContours(debug, self.polygons, -1, (0,0,255), thickness=1)

        # draw_lines(debug, self.bad_lines, thickness=2)
        #draw_lines(debug, self.vertical_lines, thickness=2, color=(255,255,0))

        draw_lines(debug, self.filtered_lines, thickness=3, color=(255,255,0))

        put_text(debug, "k=%d score: %.5f" % (self.k_means_constant, self.score), (100,100), (255, 0, 0))

        log_image(data, name, debug)

        #log_image(data, name+"_normals", self.normals_clustered)



class PipelineBarrierFinder(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.FindBarriers

    @property
    def required_keys(self) -> list:
        return ["room", "downscaled"]

    @property
    def output_keys(self) -> list:
        return []


    def run(self, data):

        self.data = data
        self.image = self.data["downscaled"]
        self.room = self.data["room"]

        diagonal = math.hypot(self.image.shape[0], self.image.shape[1])
        area = self.image.shape[0] * self.image.shape[1]

        min_line_length = diagonal / 30

        shape = (self.image.shape[1], self.image.shape[0])
        probs = resize_array(self.data["planes"]["masks"], shape)

        #obtain plane vertical lines, major barriers between original planes:
        wall = np.zeros(self.image .shape[:2], dtype=np.uint8)
        wall[self.data["isolated_labels"] == SurfaceType.Wall.index] = 1
        wall_only = wall.copy()
        for label in box_like:
            wall_only[self.data["semantic_labels"] == label.index] = 1

        on_wall = np.zeros(self.image .shape[:2], dtype=np.uint8)
        on_wall[self.data["isolated_labels"] == SurfaceType.OnWall.index] = 1
        on_wall_expanded = adjust_mask(cv2.dilate, on_wall, size=5)

        wall_like_mask = np.zeros(self.image .shape[:2], dtype=np.uint8)
        for label in box_like:
            wall_like_mask[self.data["semantic_labels"] == label.index] = 1

        wall_like_expanded = adjust_mask(cv2.dilate, wall_like_mask, size=5)

        wall[on_wall > 0] = 1

        wall[wall_like_mask > 0] = 1
        wall_contracted = adjust_mask(cv2.erode, wall, size=5)

        floor = np.zeros(self.image .shape[:2], dtype=np.uint8)
        floor[self.data["isolated_labels"] == SurfaceType.Floor.index] = 1
        floor[self.data["isolated_labels"] == SurfaceType.OnFloor.index] = 1

        ceiling = np.zeros(self.image .shape[:2], dtype=np.uint8)
        ceiling[self.data["isolated_labels"] == SurfaceType.Ceiling.index] = 1
        ceiling[self.data["isolated_labels"] == SurfaceType.OnCeiling.index] = 1
        ceiling_expanded = adjust_mask(cv2.dilate, ceiling, size=5, scale=0.5)

        on_ceiling_mask = np.zeros(self.image .shape[:2], dtype=np.uint8)
        on_ceiling_mask[self.data["isolated_labels"] == SurfaceType.OnCeiling.index] = 1
        on_ceiling_objects = [ADE20K.chandelier, ADE20K.fan] + lights
        for label in on_ceiling_objects:
            on_ceiling_mask[self.data["semantic_labels"] == label.index] = 1
        on_ceiling_mask = adjust_mask(cv2.dilate, on_ceiling_mask, size=5, scale=0.5)

        invalid_vert_areas = np.zeros(self.image .shape[:2], dtype=np.uint8)
        on_wall_objects = [ADE20K.painting, ADE20K.shelf, ADE20K.projection_screen, ADE20K.radiator, ADE20K.sconce, ADE20K.towel, ADE20K.curtain]
        for label in on_wall_objects:
            invalid_vert_areas[self.data["semantic_labels"] == label.index] = 1

        #invalid_vert_areas = adjust_mask(cv2.dilate, invalid_vert_areas, size=5, scale=0.5)

        log_mask(self.data, "invalid_vert_areas", invalid_vert_areas, self.data["downscaled"])

        index_mask = np.dstack(tuple(probs))
        index_mask = np.int32(np.argmax(index_mask, -1))

        all_planes = np.unique(index_mask).astype(np.int32)

        vertical_plane_lines = []

        #normals

        barrier_sets = []

        normals_mask = np.zeros(self.image .shape[:2], dtype=np.uint8)
        normals_mask[self.data["isolated_labels"] == SurfaceType.Wall.index] = 1
        normals_mask[self.data["isolated_labels"] == SurfaceType.OnWall.index] = 1
        normals_mask[self.data["isolated_labels"] == SurfaceType.Ceiling.index] = 1
        normals_mask[self.data["isolated_labels"] == SurfaceType.Floor.index] = 1
        normals_mask[self.data["isolated_labels"] == SurfaceType.OnFloor.index] = 1

        for label in box_like:
            normals_mask[self.data["semantic_labels"] == label.index] = 1

        #log_mask(self.data, "normals_mask", normals_mask, self.data["downscaled"])

        vps = [self.data["vertical_vp"]]
        vps.extend(self.data["horizontal_vps"])

        min_k = max(3, int(len(vps) / 3))
        if cv2.countNonZero(ceiling) > area/50:
            min_k += 1

        if cv2.countNonZero(floor) > area/50:
            min_k += 1

        max_k = min(max(min_k + 1, 2 + len(vps)), 10)     

        print("k range %d-%d" % (min_k, max_k))

        for k in range(min_k, max_k):

            vbs = VerticalBarrierSet(self.data, k, normals_mask,  wall, invalid_vert_areas)
            vbs.calculate_score()
            barrier_sets.append(vbs)

            vbs.debug(self.data, "normals_clustered_%d" % k)
            
        barriers = sorted(barrier_sets, key=lambda x: x.score, reverse=True)[0]
        barriers.calculate_score(scale=1.0)

        barriers.debug(self.data, "normals_clustered_best")

        for plane_index in all_planes:

            mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            mask[index_mask == plane_index] = 1

            total_area = cv2.countNonZero(mask)

            mask_check = mask.copy()

            mask_check[self.data["isolated_labels"] == SurfaceType.Floor.index] = 0
            mask_check[self.data["isolated_labels"] == SurfaceType.Ceiling.index] = 0
            mask_check[self.data["isolated_labels"] == SurfaceType.Other.index] = 0

            masked_area = cv2.countNonZero(mask_check)

            if masked_area < total_area / 2: continue

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            new_plane_lines = []
            for contour in contours:

                if len(contour) == 0:
                    #this may never occur, but there was a crash
                    continue

                epsilon = 5
                polygon = cv2.approxPolyDP(contour, epsilon, True)

                num_pts = len(polygon)

                for i in range(num_pts):
                    point_a = polygon[i][0]
                    point_b = polygon[(i+1) % num_pts][0]

                    length = distance.euclidean(point_a, point_b)

                    if length >= min_line_length and not line_on_image_edge(point_a, point_b, self.image.shape[1], self.image.shape[0], min_distance=diagonal/50):
                        vertical_plane_lines.append(Line(point_a[0], point_a[1], point_b[0], point_b[1], group=plane_index))

        vertical_plane_lines = list(get_inliers(vertical_plane_lines, self.data["vertical_vp"].model, np.radians(15)))
        vertical_plane_lines = merge_lines(vertical_plane_lines, search_width=max(diagonal/50, 3), search_length=1.5, angle_threshold=np.radians(20))

        def partition_lines(lines, mask=None):

            vertical_lines = list(get_inliers(lines, self.data["vertical_vp"].model, np.radians(8)))
            horizontal_lines = list(set(lines).difference(vertical_lines))

            return horizontal_lines, vertical_lines

        #grab all horizontal lines from semantic lines, which is approximately the edges of surfaces, but not between walls.
        #then also combine these with nearby horizontal lines
        horizontal_semantic_lines, vertical_semantic_lines = partition_lines(self.data["semantic_lines"])
        horizontal_semantic_lines = list(filter(lambda line: line_within_mask(line, ceiling_expanded) \
                                    and not line_within_mask(line, on_ceiling_mask) and not line_within_mask(line, wall_contracted), horizontal_semantic_lines))

        horizontal_semantic_lines = merge_lines(horizontal_semantic_lines, search_width=max(diagonal/100, 3), search_length=1.3, angle_threshold=np.radians(5))

        horizontal_vp_lines, vertical_vp_lines = partition_lines(self.data["vp_lines"])

        #all meaningful horizontal lines
        vertical_lines = vertical_semantic_lines + vertical_vp_lines

        def horizontal_line_invalid(line):
            return line_within_mask(line, wall_like_expanded) or line_within_mask(line, on_wall_expanded)

        def vertical_line_invalid(line):
            return line_within_mask(line, invalid_vert_areas)

        #re-cluster the original horizontal lines and plane vertical lines + vertical lines (do_merge=False):
        def cluster_matches(src_lines, lines, search_width, search_length=0.9, angle_threshold=np.radians(5), func_invalid=None):

            for line in src_lines + lines:
                line.cluster = None

            cluster_index = 0

            for src_line in src_lines:
                src_line.cluster = cluster_index

                src_rect = src_line.bounding_box(width=search_width, length_multiplier=search_length)

                for line in lines:

                    if func_invalid is not None and func_invalid(line):
                        continue

                    #maybe use vanishing points instead?
                    if LineFunctions.line_angle_difference(src_line.angle, line.angle) > angle_threshold:
                        continue

                    rect = line.bounding_box(width=3)

                    result, _ = cv2.rotatedRectangleIntersection(src_rect, rect)

                    if result != 0:
                        line.cluster = cluster_index

                cluster_index += 1

        cluster_matches(horizontal_semantic_lines, horizontal_vp_lines, diagonal / 15, func_invalid=horizontal_line_invalid)

        #find intersections of grouped lines, being careful not to corrupt originals (copy)

        #extend these merged horizontal lines together to find intersections 
        
        #horizontal_semantic_lines = list(filter(lambda line: line.length > diagonal / 40, horizontal_semantic_lines))

        horizontal_clusters = list(set([line.cluster for line in horizontal_semantic_lines]))
        adjacent_horizontal_lines = list(filter(lambda line: line.cluster in horizontal_clusters and line not in horizontal_semantic_lines, horizontal_vp_lines))

        debug = self.data["downscaled"].copy()
        draw_lines(debug, adjacent_horizontal_lines, thickness=2, color=(0,255,255))
        draw_lines(debug, horizontal_semantic_lines, thickness=2, color=(255,255,0))
        log_image(self.data, "barriers_unfiltered", debug)

        #extend lines into adjacent lines
        search_width = diagonal / 300
        angle_threshold = np.radians(3)
        horizontal_lines = horizontal_semantic_lines + adjacent_horizontal_lines

        for i, line_a in enumerate(horizontal_lines):

            if line_a.dead: continue

            min_index = i

            line_data = line_a.data.copy()
            rect_a = line_a.bounding_box(width=search_width, length_offset=diagonal)

            for j, line_b in enumerate(horizontal_lines):

                if line_b.dead or LineFunctions.line_angle_difference(line_a.angle, line_b.angle) > angle_threshold or line_a == line_b:
                    continue

                rect_b = line_b.bounding_box(width=search_width, length_offset=diagonal)

                result, intersections = cv2.rotatedRectangleIntersection(rect_a, rect_b)

                if result == 0: continue

                point = np.mean(intersections, axis=0)[0]

                distance_between = min(distance.euclidean(line_a.point_a, line_b.point_a), distance.euclidean(line_a.point_b, line_b.point_b), distance.euclidean(line_a.point_a, line_b.point_b))
                
                min_distance_a = min(distance.euclidean(point, line_a.point_a), distance.euclidean(point, line_a.point_b))
                min_distance_b = min(distance.euclidean(point, line_b.point_a), distance.euclidean(point, line_b.point_b))

                #make sure it extends to the point
                if distance_between < (2 + min(min_distance_a, min_distance_b)): continue

                line_a.dead = True
                line_b.dead = True
                min_index = min(min_index, j)

                line_data = LineFunctions.merge_line_pair(line_data[0], line_data[1], line_data[2], line_data[3], \
                                                         line_b.data[0], line_b.data[1], line_b.data[2], line_b.data[3], \
                                                         line_b.dx, line_b.dy)

            if line_a.dead:
                horizontal_lines[min_index] = Line(line_data[0], line_data[1], line_data[2], line_data[3], cluster=line_a.cluster)

        #merge conventionally, do not filter out .dead
        horizontal_lines = merge_lines(horizontal_lines, search_width=diagonal/300, search_length=1.1, angle_threshold=np.radians(5), do_filter=False)

        split_index = len(horizontal_semantic_lines)

        horizontal_semantic_lines = list(filter(lambda x: not x.dead, horizontal_lines[:split_index]))
        adjacent_horizontal_lines = list(filter(lambda x: not x.dead, horizontal_lines[split_index:]))

        split_index = len(horizontal_semantic_lines)

        horizontal_lines = adjacent_horizontal_lines + horizontal_semantic_lines
        intersections = extend_to_intersection(horizontal_lines, search_length=2.0, modify=True)

        horizontal_semantic_lines = horizontal_lines[:split_index]
        adjacent_horizontal_lines = horizontal_lines[split_index:]

        cluster_matches(vertical_plane_lines, vertical_lines, diagonal/20, angle_threshold=np.radians(20), func_invalid=vertical_line_invalid)
        vertical_clusters = list(set([line.cluster for line in vertical_plane_lines]))

        adjacent_vertical_lines = list(filter(lambda line: line.cluster in vertical_clusters, vertical_lines))

        horizontal_lines = list(set(horizontal_semantic_lines + adjacent_horizontal_lines))

        self.data["horizontal_barriers"] = horizontal_lines

        self.data["vertical_barriers"] = adjacent_vertical_lines

        samples = []
        for intersection in intersections:
            if intersection.term_a:
                samples.append(intersection.term_a.point)

            if intersection.term_b:
                samples.append(intersection.term_b.point)


        if len(samples) > 0:
            samples = np.unique(np.array(samples), axis=0)

        results = []

        for k in range(2, len(samples)):

            kmeans = KMeans(n_clusters=k, random_state=0).fit(samples)
            avg_distance = np.sqrt(kmeans.inertia_ / len(samples))


            labels = kmeans.labels_
            centers = kmeans.cluster_centers_

            sil_coeff = silhouette_score(samples, labels, metric='euclidean')

            print("k=%d" % k, labels, avg_distance, sil_coeff)

            results.append((sil_coeff, avg_distance, labels, centers))


        if len(results) > 1:
            index = np.argmax(np.array(results)[:,0])
            best_result = results[index]
            print("BEST", best_result)
            centers = best_result[3]
        else:
            centers = samples

        if im_logging_enabled(data):
            #debug = get_segmentation_image(self.room.index_mask, self.data["downscaled"], labelset=None)\
            debug = self.data["downscaled"].copy()


            #debug = cv2.addWeighted(debug, 0.7, self.data["downscaled"], 0.3, 0)

            draw_lines(debug, self.data["vp_lines"], color=(50, 50, 50), thickness=1)

            draw_lines(debug, vertical_plane_lines, color=(0,255,0), thickness=3)
            draw_lines(debug, adjacent_vertical_lines, color=(255, 0, 0), thickness=2)
            
            draw_lines(debug, adjacent_horizontal_lines, color=(0, 255, 255), thickness=2)
            draw_lines(debug, horizontal_semantic_lines, color=(255,255,0), thickness=2)

            #draw_lines(debug, linked_horizontal_lines, color=(50,255,255), thickness=2)

            def constrain_point(point):
                x = int(min(max(point[0], 0), debug.shape[1]))
                y = int(min(max(point[1], 0), debug.shape[0]))
                return x, y

            for intersection in intersections:

                def debug_term(term):
                    if term is None: return
                    cv2.circle(debug, constrain_point(term.point), 5, (255, 180, 0), cv2.FILLED, cv2.LINE_AA)
                    draw_lines(debug, [term.line], thickness=2)
                
                draw_lines(debug, [intersection.line], thickness=2)

                debug_term(intersection.term_a)
                debug_term(intersection.term_b)

            for center in centers:
                cv2.circle(debug, constrain_point(center), 8, (0, 255, 0), 2)
                
            log_image(self.data, "barriers", debug)

