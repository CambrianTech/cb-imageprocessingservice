import numpy as np
import math
import cv2
import sys
import random
from scipy.spatial import distance

import pyximport; pyximport.install(language_level=3)
from LineFunctions import LineFunctions

class Line:

    def __init__(self, x0, y0, x1, y1, contour_group=-1, contour_index=-1):
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
    def merge(cls, line_data, search_width, search_length, angle_threshold, max_color_std=None, min_confidence=0,
              length_offset=0, color=None):

        initial_count = len(line_data)
        print("Merging %d lines" % (initial_count))

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
                                # data = (line_b.point_a, line_b.point_b)

            if len(parallel_lines) > 0:
                new_line = Line(data[0][0], data[0][1], data[1][0], data[1][1])
                line_a.dead = True
                for line in parallel_lines:
                    line.dead = True

                if color is not None:
                    new_line.color = color

                line_data.append(new_line)

            # end while

        # clean up dead lines
        line_data = list(filter(lambda x: not x.dead, line_data))
        if min_confidence > 0:
            line_data = list(filter(lambda x: x.get_confidence() >= min_confidence, line_data))

        print("Reduced lines by %d" % (initial_count - len(line_data)))

        return line_data

                # if angle_diff <= .001:
                #     max_length = line_b.length
                #     max_confidence = line_a.get_confidence()
                #     data = LineFunctions.merge_lines(data, (line_b.point_a, line_b.point_b))
                #
                #     parallel_lines.append(line_b)
                    # l2 = line_b.length**2
                    # p1 = np.float32(line_b.point_a)
                    # p2 = np.float32(line_b.point_b)
                    # p3 = np.float32(line_a.point_a)
                    # p4 = np.float32(line_a.point_b)
                    # if l2 == 0:
                    #     print('p1 and p2 are the same points')
                    #     continue
                    #
                    # t1 = np.sum((p3 - p1) * (p2 - p1)) / l2
                    # t2 = np.sum((p4 - p1) * (p2 - p1)) / l2
                    #
                    # # if abs(t1) >2.0 or abs(t2) > 2.0:
                    # #     continue
                    #
                    # pr1 = p1 + t1 * (p2 - p1)
                    # pr2 = p1 + t2 * (p2 - p1)
                    # d = distance.euclidean(pr1,p1)/2.+distance.euclidean(pr2,p2/2.)
                    #
                    # if d>100: continue
                    # print("distance", d)
                    # pts = np.float32([p1, pr1, p2, pr2])
                    #
                    # pta = pts[np.argmin(pts[:, 0])]
                    # ptb = pts[np.argmax(pts[:, 0])]

                    # data = ((pta[0],pta[1]),(ptb[0],ptb[1]))


                    # print(data)


                    # rect_b = line_b.bounding_box(2, length_multiplier=search_length, length_offset=length_offset)
                    # result, intersections = cv2.rotatedRectangleIntersection(rect_a, rect_b)
                    #
                    # if not intersections is None:
                    #     std = line_a.get_color_std() + 0.1
                    #
                    #     if max_color_std is not None:
                    #         diff = np.abs(line_a.get_color_mean() - line_b.get_color_mean())
                    #         max_diff = np.mean(diff / std)
                    #
                    #     if max_color_std is None or max_diff <= max_color_std:
                    #         parallel_lines.append(line_b)
                    #         data = LineFunctions.merge_lines(data, (line_b.point_a, line_b.point_b))
                    #
                    #         if line_b.length > 0.8 * max_length and line_b.get_confidence() > max_confidence:
                    #             max_length = line_b.length
                    #             max_confidence = line_b.get_confidence()
                    #             radius = distance.euclidean(data[0], data[1]) / 2
                    #             midpoint = ((data[0][0] + data[1][0]) / 2, (data[0][1] + data[1][1]) / 2)
                    #             dx = math.cos(line_b.angle) * radius
                    #             dy = math.sin(line_b.angle) * radius
                    #             data = ((midpoint[0] + dx, midpoint[1] + dy), (midpoint[0] - dx, midpoint[1] - dy))
                    #             # data = (line_b.point_a, line_b.point_b)

            #
            #
            # if len(parallel_lines) > 0:
            #     new_line = Line(data[0][0], data[0][1], data[1][0], data[1][1])
            #     line_a.dead = True
            #     for line in parallel_lines:
            #         line.dead = True
            #
            #     if color is not None:
            #         new_line.color = color
            #     line_data.append(new_line)


            #end while
        #
        # #clean up dead lines
        # line_data = list(filter(lambda x: not x.dead, line_data))
        #
        # if min_confidence > 0:
        #     line_data = list(filter(lambda x: x.get_confidence() >= min_confidence, line_data))
        #
        # print("Reduced lines by %d" % (initial_count - len(line_data)))
        #
        #
        # return line_data
        # initial_count = len(line_data)
        # print("Merging %d lines" % (initial_count))
        # min_confidence=.5
        # i = 0
        # while i < len(line_data):
        #     line_a = line_data[i]
        #     i += 1
        #
        #     if line_a.dead or (min_confidence > 0 and line_a.get_confidence() < min_confidence): continue
        #
        #     rect_a = line_a.bounding_box(search_width, length_multiplier=search_length, length_offset=length_offset)
        #     data = (line_a.point_a, line_a.point_b)
        #
        #     parallel_lines = []
        #     max_length = line_a.length
        #     max_confidence = line_a.get_confidence()
        #     print(line_a.get_confidence())
        #     # continue
        #     for j in range(i+1, len(line_data)):
        #         continue
        #         line_b = line_data[len(line_data)-j]
        #
        #         if line_b.dead or (min_confidence > 0 and line_b.get_confidence() < min_confidence): continue
        #
        #         angle_diff = LineFunctions.line_angle_difference(line_a.angle, line_b.angle)
        #
        #         if angle_diff <= angle_threshold:
        #             rect_b = line_b.bounding_box(2, length_multiplier=search_length, length_offset=length_offset)
        #             result, intersections = cv2.rotatedRectangleIntersection(rect_a, rect_b)
        #
        #             if not intersections is None:
        #                 std = line_a.get_color_std() + 0.1
        #
        #                 if max_color_std is not None:
        #                     diff = np.abs(line_a.get_color_mean() - line_b.get_color_mean())
        #                     max_diff = np.mean(diff / std)
        #
        #                 if max_color_std is None or max_diff <= max_color_std:
        #
        #                     # data = LineFunctions.merge_lines(data, (line_b.point_a, line_b.point_b))
        #                     # new_conf = Line(n_data[0][0], n_data[0][1], n_data[1][0], n_data[1][1]).get_confidence()
        #                     # # if new_conf>max(max_confidence,line_b.get_confidence()):
        #                     # data=n_data
        #                     # max_confidence=new_conf
        #                     # if line_b.get_confidence()<.9:
        #                         # parallel_lines.append(line_b)
        #                     # print("c", new_conf, line_b.get_confidence())
        #
        #                     if line_b.get_confidence() > line_a.get_confidence():
        #
        #                         # max_length = line_b.length
        #                         max_confidence = line_b.get_confidence()
        #
        #                         l2 =line_b.length
        #                         p1 = np.float32(line_b.point_a)
        #                         p2 = np.float32(line_b.point_b)
        #                         p3 = np.float32(line_a.point_a)
        #                         p4 = np.float32(line_a.point_b)
        #                         if l2 == 0:
        #                             print('p1 and p2 are the same points')
        #
        #                             t1 = np.sum((p3 - p1) * (p2 - p1)) / l2
        #                             t2 = np.sum((p4 - p1) * (p2 - p1)) / l2
        #                             if abs(t1)<5 and abs(t2)<5:
        #
        #                                 pr1 = p1 + t1 * (p2 - p1)
        #                                 pr2 = p1 + t2 * (p2 - p1)
        #
        #                                 pts = np.float32([p1,pr1,p2,pr2])
        #
        #                                 # data = (pts[np.argmin(pts[:,0])], pts[np.argmax(pts[:,0])])
        #                         # print("d",np.round(distance.euclidean(data[0],data[1])-l2))
        #                         #                            if line_b.get_confidence() > max_confidence:
        #                     # else:
        #                     #
        #                     #     l2 =line_a.length
        #                     #     p3 = np.float32(line_b.point_a)
        #                     #     p4 = np.float32(line_b.point_b)
        #                     #     p1 = np.float32(line_a.point_a)
        #                     #     p2 = np.float32(line_a.point_b)
        #                     #     if l2 == 0:
        #                     #         print('p1 and p2 are the same points')
        #                     #
        #                     #     t1 = np.sum((p3 - p1) * (p2 - p1)) / l2
        #                     #     t2 = np.sum((p4 - p1) * (p2 - p1)) / l2
        #                     #     if abs(t1)>100 or abs(t2)>100:
        #                     #         # print("t1", t1, t2)
        #                     #         continue
        #                     #     pr1 = p1 + t1 * (p2 - p1)
        #                     #     pr2 = p1 + t2 * (p2 - p1)
        #
        #                         # pts = np.float32([p1,pr1,p2,pr2])
        #                         # print(np.int32(pts))
        #                         # print("p", np.int32([p1, p2]), np.int32([p3, p4]),np.int32([pr1, pr2]))
        #
        #                         # line_a = (pts[np.argmin(pts[:,0])], pts[np.argmax(pts[:,0])])
        #
        #
        #     if len(parallel_lines) > 0:
        #         new_line = Line(data[0][0], data[0][1], data[1][0], data[1][1])
        #         # line_a.dead = True
        #         for line in parallel_lines:
        #             line.dead = True
        #
        #         if color is not None:
        #             new_line.color = color
        #
        #         line_data.append(new_line)
        #
        #
        #     #end while
        #
        # #clean up dead lines
        # line_data = list(filter(lambda x: not x.dead, line_data))
        # if min_confidence > 0:
        #     line_data = list(filter(lambda x: x.get_confidence() >= min_confidence, line_data))
        #
        # print("Reduced lines by %d" % (initial_count - len(line_data)))
        #
        #
        # return line_data

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
            return (255,255,255) if self.color is None else self.color
        elif self.confidence > 0.80:
            return (0,255,255)
        elif self.confidence > 0.5:
            return (0,127,255)
        elif self.confidence > 0.3:
            return (255,0,255)
        elif self.confidence > 0.1:
            return (255,0,0)
        else:
                return (0,0,255)

    def draw(self, image, color=None, thickness=None, lineType=cv2.LINE_8, scale_x=1.0, scale_y=1.0):
        if color is None:
            color = self.confidence_color()
        if thickness is None:
            thickness = max(int(self.diagonal / 600), 2)

        # if self.source_lines is not None:
        #     color = (255,100,0)

        pt1 = (int(self.point_a[0]*scale_x), int(self.point_a[1]*scale_y))
        pt2 = (int(self.point_b[0]*scale_x), int(self.point_b[1]*scale_y))

        cv2.line(image,pt1, pt2, color, thickness, lineType=lineType)

    @classmethod
    def draw_all(cls, line_data, image, color=None, thickness=None, lineType=cv2.LINE_8, sx=1.0, sy=1.0):
        for line in line_data:
            line.draw(image, color, thickness, lineType=lineType, scale_x=sx, scale_y=sy)
        return image
    
