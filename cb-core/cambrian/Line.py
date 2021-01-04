import numpy as np
import math
import cv2
import sys
import random
from scipy.spatial import distance

import pyximport; pyximport.install(language_level=3)
from cambrian.LineFunctions import LineFunctions

class Line:

    def __init__(self, x0, y0, x1, y1, contour_group=-1, contour_index=-1, source_lines=None):
        self.point_a = (x0, y0)
        self.point_b = (x1, y1)

        self.contour_group = contour_group
        self.contour_index = contour_index

        self.dead = False

        self.samples = None
        self.confidence = None
        self.color_mean = None
        self.color_std = None
        self.color = None
        self.source_lines = source_lines

        self.recalculate()

    def recalculate(self):
        self.midpoint = ((self.point_a[0] + self.point_b[0]) / 2, (self.point_a[1] + self.point_b[1]) / 2)
        self.length_sq = distance.sqeuclidean(self.point_a, self.point_b)
        self.length = math.sqrt(self.length_sq)
        self.angle = LineFunctions.line_angle(self.point_a[0], self.point_a[1], self.point_b[0], self.point_b[1])

    #class variables:
    image = None
    edges = None
    debug = None
    confidence_step = 8
    color_step = 7

    @classmethod
    def prepare(cls, image, edges, debug):
        cls.image = image
        cls.edges = edges
        cls.debug = debug
        cls.diagonal = math.hypot(cls.image.shape[0], cls.image.shape[1])

    def get_confidence(self):
        if self.confidence is None:
            self.confidence = LineFunctions.line_contour_confidence(self.point_a, self.point_b, self.edges, int(self.length / self.confidence_step) + 1)
        return self.confidence

    def get_samples(self):
        if self.samples is None:
            self.samples = LineFunctions.get_line_samples(self.point_a, self.point_b, self.image, int(self.length / self.color_step) + 1)
        return self.samples

    def find_line_pair(self, min_width, min_length=0.3, angle_diff=math.radians(5)):
        points = []
        for line in self.source_lines:
            ratio = min(self.length / line.length, line.length / self.length)
            if ratio >= min_length:
                points.append((int(line.point_a[0]), int(line.point_a[1])))
                points.append((int(line.point_b[0]), int(line.point_b[1])))

        if len(points) < 4:
            return None

        hull = cv2.convexHull(np.array(points))

        if len(hull) < 4:
            return None

        area = cv2.contourArea(hull)
        width = area / self.length

        if width < min_width:
            return None


        hull = hull.reshape(len(hull), 2)

        lines = []
        for i in range(len(hull)):
            point_a = hull[i]
            point_b = hull[(i+1) % len(hull)]
            new_line = Line(point_a[0], point_a[1], point_b[0], point_b[1])

            if LineFunctions.line_angle_difference(self.angle, new_line.angle) < angle_diff:
                lines.append(new_line)
            
        return lines if len(lines) > 1 else None

    def get_color_mean(self):
        if self.color_mean is None:
            self.color_mean = np.mean(self.get_samples(), axis=0)
        return self.color_mean

    def get_color_std(self):
        if self.color_std is None:
            self.color_std = np.std(self.get_samples(), axis=0)
        return self.color_std

    @classmethod
    def _out_of_range(cls, point):
        return point[0] < 0 or point[1] < 0 or point[0] >= cls.image.shape[1] or point[1] >= cls.image.shape[0]

    def bounding_box(self, width=10, length_multiplier=1.0, length_offset=0.0):
        return (self.midpoint, (max(self.length * length_multiplier, self.length + length_offset), width), np.degrees(self.angle))

    @classmethod
    def merge(cls, line_data, search_width, search_length, angle_threshold, \
        max_color_std=None, min_confidence=0, length_offset=0, create_pairs=False, min_pair_width=None):

        initial_count = len(line_data)
        print("Merging %d lines" % (initial_count))

        if min_pair_width is None:
            min_pair_width = cls.diagonal/250

        i = 0
        while i < len(line_data):
            line_a = line_data[i]
            i += 1

            if line_a.dead or (min_confidence > 0 and line_a.get_confidence() < min_confidence): continue

            rect_a = line_a.bounding_box(search_width, length_multiplier=search_length, length_offset=length_offset)
            data = (line_a.point_a, line_a.point_b)

            parallel_lines = []
            max_length = line_a.length
            max_confidence = line_a.get_confidence()

            for j in range(i, len(line_data)):

                line_b = line_data[j]

                if line_b.dead or (min_confidence > 0 and line_b.get_confidence() < min_confidence): continue

                angle_diff = LineFunctions.line_angle_difference(line_a.angle, line_b.angle)

                if angle_diff <= angle_threshold:
                    rect_b = line_b.bounding_box(2, length_multiplier=search_length, length_offset=length_offset)
                    result, intersections = cv2.rotatedRectangleIntersection(rect_a, rect_b)

                    if not intersections is None:
                        std = line_a.get_color_std() + 0.1

                        if max_color_std is not None:
                            diff = np.abs(line_a.get_color_mean() - line_b.get_color_mean())
                            max_diff = np.mean(diff / std)

                        if max_color_std is None or max_diff <= max_color_std:
                            parallel_lines.append(line_b)
                            data = LineFunctions.merge_lines(data, (line_b.point_a, line_b.point_b))

                            if line_b.length > 0.8 * max_length and line_b.get_confidence() > max_confidence:
                                max_length = line_b.length
                                max_confidence = line_b.get_confidence()
                                radius = distance.euclidean(data[0], data[1]) / 2
                                midpoint = ((data[0][0] + data[1][0]) / 2, (data[0][1] + data[1][1]) / 2)
                                dx = math.cos(line_b.angle) * radius
                                dy = math.sin(line_b.angle) * radius
                                data = ((midpoint[0] + dx, midpoint[1] + dy), (midpoint[0] - dx, midpoint[1] - dy))
                                #data = (line_b.point_a, line_b.point_b)
                                
                            

            if len(parallel_lines) > 0:
                parallel_lines.append(line_a)
                source_lines = parallel_lines
                new_line = Line(data[0][0], data[0][1], data[1][0], data[1][1])

                for line in parallel_lines:
                    if line.source_lines is not None:
                        source_lines.extend(line.source_lines)
                    line.dead = True

                if create_pairs:
                    new_line.source_lines = source_lines

                line_data.append(new_line)

            #end while

        #clean up dead lines
        line_data = list(filter(lambda x: not x.dead, line_data))
        if min_confidence > 0:
            line_data = list(filter(lambda x: x.get_confidence() >= min_confidence, line_data))

        #now get hull of parallel
        if create_pairs:
            parallel_lines = list(filter(lambda x: x.source_lines is not None, line_data))
            for line in parallel_lines:
                pair = line.find_line_pair(min_pair_width)
                if pair is not None:
                    line.dead = True
                    line_data.extend(pair)

        print("Reduced lines by %d" % (initial_count - len(line_data)))

        line_data = list(filter(lambda x: not x.dead, line_data))

        return line_data

    @classmethod
    def shift_lines(cls, line_data, distance):
        for line in line_data:

            orthagonal_angle = line.angle + math.pi/2
            best_score = 0
            best_line = line
            hits = 0

            max_value = int(max(np.round(distance), 2))

            for length in range(1, max_value):

                dx = length * math.cos(orthagonal_angle)
                dy = length * math.sin(orthagonal_angle)

                up_point_a = (line.point_a[0] + dx, line.point_a[1] + dy)
                up_point_b = (line.point_b[0] + dx, line.point_b[1] + dy)

                down_point_a = (line.point_a[0] - dx, line.point_a[1] - dy)
                down_point_b = (line.point_b[0] - dx, line.point_b[1] - dy)

                num_samples = 1 + int(line.length/5)
                up_conf = LineFunctions.line_contour_confidence(up_point_a, up_point_b, cls.crisp_edges, num_samples)
                down_conf = LineFunctions.line_contour_confidence(down_point_a, down_point_b, cls.crisp_edges, num_samples)

                if up_conf > best_score or down_conf > best_score:
                    hits += 1
                    if up_conf > best_score:
                        best_score = up_conf
                        best_line = (up_point_a, up_point_b)

                    if down_conf > best_score:
                        best_score = down_conf
                        best_line = (down_point_a, down_point_b)

                if hits > 2: break

            if best_line != line:
                line.point_a = best_line[0]
                line.point_b = best_line[1]
                line.recalculate()
                #print("shift by ", min(cls._euclidean_dist(line.point_a, best_line[0]), cls._euclidean_dist(line.point_b, best_line[0])))

        return line_data

    @classmethod
    def merge_corners(cls, line, line_data, corners, corner_intersections, max_color_std, confidence_diff):
        corner_a = None
        corner_a_index = -1
        best_a_corner_dist = sys.maxsize

        corner_b = None
        corner_b_index = -1
        best_b_corner_dist = sys.maxsize

        min_sample_gap = cls.diagonal/60

        for corner, is_parallel, index in corner_intersections:

            a_dist = distance.sqeuclidean(line.point_a, corner)
            b_dist = distance.sqeuclidean(line.point_b, corner)

            if a_dist < b_dist:
                if corner_a is None or a_dist < best_a_corner_dist:
                    corner_a = corner
                    corner_a_index = index if is_parallel else -1
                    best_a_corner_dist = a_dist
            else:
                if corner_b is None or b_dist < best_b_corner_dist:
                    corner_b = corner
                    corner_b_index = index if is_parallel else -1
                    best_b_corner_dist = b_dist

        if corner_a is not None or corner_b is not None:
            line_mean = line.get_color_mean()
            line_std = line.get_color_std()
            line_confidence = line.get_confidence()
            num_line_samples = int(line.length / cls.color_step) + 1
            num_line_conf_samples = int(line.length / cls.confidence_step) + 1

            ab_distance = max(cls.diagonal / 400, 1)
            above, below = LineFunctions.get_parallel_lines(line.point_a, line.point_b, line.length, ab_distance) 
            #cv2.line(cls.debug, (int(above[0][0]), int(above[0][1])), (int(above[1][0]), int(above[1][1])), (255,0,255), 1)

            above_color = np.mean(LineFunctions.get_line_samples(above[0], above[1], cls.image, num_line_samples), axis=0)
            below_color = np.mean(LineFunctions.get_line_samples(below[0], below[1], cls.image, num_line_samples), axis=0)
            ab_diff = above_color - below_color

            def is_match(point, corner, is_point_a):
                length = distance.euclidean(point, corner)
                if length < max(min_sample_gap, 3): return True

                num_color_samples = int(min(num_line_samples, length+1))

                if num_color_samples > 3:
                    extension_samples = LineFunctions.get_line_samples(point, corner, cls.image, num_color_samples)
                    extension_color = np.mean(extension_samples, axis=0)
                    diff = np.abs(extension_color - line_mean)
                    mean_diff = np.mean(diff / (line_std + 0.1))

                    if mean_diff > max_color_std: return False

                    extension_std = np.std(extension_samples, axis=0)
                    std_diff = np.max(np.abs(extension_std - line_std))

                    #todo: need more comparisons to prevent false positives:
                    if std_diff > 10: return False

                    # #compare correct order
                    above_e, below_e = LineFunctions.get_parallel_lines(corner, point, length, ab_distance) if is_point_a else LineFunctions.get_parallel_lines(point, corner, length, ab_distance)
                    #cv2.line(cls.debug, (int(above_e[0][0]), int(above_e[0][1])), (int(above_e[1][0]), int(above_e[1][1])), (0,0,0), 1)

                    above_color_e = np.mean(LineFunctions.get_line_samples(above_e[0], above_e[1], cls.image, num_line_samples), axis=0)
                    below_color_e = np.mean(LineFunctions.get_line_samples(below_e[0], below_e[1], cls.image, num_line_samples), axis=0)
                    ab_diff_e = above_color_e - below_color_e

                    above_diff = np.max(np.abs(above_color_e - above_color))
                    below_diff = np.max(np.abs(below_color_e - below_color))
                    ab_diff_diff = np.max(np.abs(ab_diff_e - ab_diff))

                    if ab_diff_diff > 20: return False

                    if above_diff > 60 or below_diff > 60: return False


                num_conf_samples = int(min(num_line_conf_samples, length+1))

                if num_conf_samples > 3:
                    extension_confidence = LineFunctions.line_contour_confidence(point, corner, cls.edges, num_conf_samples)
                    if abs(extension_confidence - line_confidence) > confidence_diff: return False

                return True #additional similarity check

            #line point a match:
            if not corner_a is None:
                if is_match(line.point_a, corner_a, True):
                    if corner_a_index >= 0: #parallel
                        line.point_a, line.point_b = LineFunctions.merge_lines((line.point_a, line.point_b), (line_data[corner_a_index].point_a, line_data[corner_a_index].point_b))
                        line_data[corner_a_index].dead = True
                    else:
                        line.point_a = corner_a
                    corners.append(corner_a)
                #else: print("rejected")

            #line point b match:
            if not corner_b is None:
                if is_match(line.point_b, corner_b, False):
                    if corner_b_index >= 0:
                        line.point_a, line.point_b = LineFunctions.merge_lines((line.point_a, line.point_b), (line_data[corner_b_index].point_a, line_data[corner_b_index].point_b))
                        line_data[corner_b_index].dead = True
                    else:
                        line.point_b = corner_b
                    corners.append(corner_b)
                #else: print("rejected")

            line.recalculate()

        return line

    @classmethod
    def find_corners(cls, line_data, search_length, angle_threshold, parallel_threshold=math.radians(3.0), max_color_std=20, confidence_diff=0,  min_length=0):

        corners = []
        orthagonal_threshold = math.radians(10.0)
        length_offset = cls.diagonal / 50

        for i in range(len(line_data)):
            line_a = line_data[i]

            if line_a.dead or line_a.length < min_length: continue

            corner_intersections = []

            rect_a = line_a.bounding_box(3, length_multiplier=search_length, length_offset=length_offset)

            for j in range(len(line_data)):
                if i == j: continue

                line_b = line_data[j]
                if line_b.dead or line_b.length < min_length: continue

                angle_diff = LineFunctions.line_angle_difference(line_a.angle, line_b.angle)
                is_parallel = angle_diff <= parallel_threshold
                is_orthagonal = abs(angle_diff - math.pi/2) < orthagonal_threshold

                if angle_diff >= angle_threshold or is_parallel:

                    rect_b = line_b.bounding_box(3, length_offset=length_offset)
                    _, intersections = cv2.rotatedRectangleIntersection(rect_a, rect_b)
                    
                    if intersections is not None:
                        if is_parallel:
                            #if it overlaps it is invalid:
                            _, _intersections = cv2.rotatedRectangleIntersection(line_a.bounding_box(3), line_b.bounding_box(3))
                            if _intersections is not None: continue

                        #precise intersection
                        intersection = np.ravel(np.mean(intersections, axis=0))
                        #intersection = LineFunctions.get_intersection(line_a.point_a, line_a.point_b, line_b.point_a, line_b.point_b)

                        if intersection is None or max(intersection[0], intersection[1]) > cls.diagonal * 2: continue

                        new_half_length = distance.euclidean(line_a.midpoint, intersection)

                        if (not is_parallel and new_half_length >= 0.45 * line_a.length) or (is_parallel and new_half_length >= 0.51 * line_a.length):
                            corner_intersections.append((intersection, is_parallel, j))
                            if cls.debug is not None:
                                cv2.circle(cls.debug, (int(intersection[0]), int(intersection[1])), max(int(cls.diagonal/200), 3), (0,0,255), 2)

            if len(corner_intersections) > 0:
                cls.merge_corners(line_data[i], line_data, corners, corner_intersections, max_color_std, confidence_diff)
                
            #end while

        #clean up dead lines
        print("Found %d corners" % len(corners))
        line_data = list(filter(lambda x: not x.dead, line_data))

        return line_data, corners

    def confidence_color(self):
        if self.confidence is None:
            return (255,0,200) if self.color is None else self.color
        elif self.confidence > 0.80:
            return (0,255,0)
        elif self.confidence > 0.5:
            return (0,220,150)
        elif self.confidence > 0.3:
            return (0,255,255)
        elif self.confidence > 0.1:
            return (0,100,255)
        else:
            return (0,0,255)

    def draw(self, image, color=None, thickness=None):
        if color is None:
            color = self.confidence_color()
        if thickness is None:
            thickness = max(int(self.diagonal / 600), 2)

        # if self.source_lines is not None:
        #     color = (255,100,0)

        cv2.line(image, (int(self.point_a[0]), int(self.point_a[1])), (int(self.point_b[0]), int(self.point_b[1])), color, thickness)

    @classmethod
    def draw_all(cls, line_data, image, color=None, thickness=None):
        for line in line_data:
            line.draw(image, color, thickness)
        return image

    @classmethod
    def compute_edgelets(cls, line_data, class_labels=None):
        locations = []
        directions = []
        strengths = []
        classes = []

        for line in line_data:
            p0, p1 = np.array([line.point_a[0], line.point_a[1]]), np.array([line.point_b[0], line.point_b[1]])
            if class_labels is not None:
                classes.append(class_labels[int(line.midpoint[1]), int(line.midpoint[0])])

            locations.append(line.midpoint)
            directions.append(p1 - p0)
            strengths.append(line.length)

        locations = np.array(locations)
        directions = np.array(directions)
        strengths = np.array(strengths)
        classes = np.array(classes)

        directions = np.array(directions) / np.linalg.norm(directions, axis=1)[:, np.newaxis]

        return (locations, directions, strengths, classes)
    
