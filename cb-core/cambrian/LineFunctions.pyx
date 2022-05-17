import numpy as np
import math
import cv2
import sys
import random
from scipy.spatial import distance

from libc.math cimport abs, sqrt, atan, atan2, pi, sin, cos

cdef double pi_2 = pi * 0.5

cdef tuple _get_parallel_lines(double point_a_x, double point_a_y, double point_b_x, double point_b_y, double length, double distance):

    cdef double dx = (point_b_x - point_a_x) / length
    cdef double dy = (point_b_y - point_a_y) / length

    cdef double up_point_a_x = point_a_x + dy * distance
    cdef double up_point_a_y = point_a_y + dx * distance

    cdef double up_point_b_x = point_b_x + dy * distance
    cdef double up_point_b_y = point_b_y + dx * distance

    cdef double down_point_a_x = point_a_x - dy * distance
    cdef double down_point_a_y = point_a_y - dx * distance

    cdef double down_point_b_x = point_b_x - dy * distance
    cdef double down_point_b_y = point_b_y - dx * distance

    return ((up_point_a_x, up_point_a_y), (up_point_b_x, up_point_b_y)), ((down_point_a_x, down_point_a_y), (down_point_b_x, down_point_b_y))

cdef tuple _get_intersection(tuple line_a_a, tuple line_a_b, tuple line_b_a, tuple line_b_b):
    # a1x + b1y = c1
    cdef double a1 = line_a_b[1] - line_a_a[1]
    cdef double b1 = line_a_a[0] - line_a_b[0]
    cdef double c1 = a1 * (line_a_a[0]) + b1 * (line_a_a[1])

    # a2x + b2y = c2
    cdef double a2 = line_b_b[1] - line_b_a[1]
    cdef double b2 = line_b_a[0] - line_b_b[0]
    cdef double c2 = a2 * (line_b_a[0]) + b2 * (line_b_a[1])

    cdef double det = a1 * b2 - a2 * b1

    if abs(det) <= 0.001:
        return (-1,-1)

    x = ((b2 * c1) - (b1 * c2)) / det
    y = ((a1 * c2) - (a2 * c1)) / det
    return (x, y)

#from opencv FastLineDetectorImpl::mergeLines
cdef tuple _merge_line_pair(ax, ay, bx, by, cx, cy, dx, dy, dljx, dljy):
    
    cdef float thi; 
    cdef float thj; 
    cdef float thr;

    cdef float dlix = (bx - ax);
    cdef float dliy = (by - ay);
    # cdef float dljx = (dx - cx);
    # cdef float dljy = (dy - cy);

    cdef float li = sqrt((dlix * dlix) + (dliy * dliy));
    cdef float lj = sqrt((dljx * dljx) + (dljy * dljy));

    cdef float xg = (li * (ax + bx) + lj * (cx + dx)) / (2.0 * (li + lj));
    cdef float yg = (li * (ay + by) + lj * (cy + dy)) / (2.0 * (li + lj));

    if (dlix == 0.0): thi = pi_2;
    else: thi = atan(dliy / dlix);

    if (dljx == 0.0): thj = pi_2;
    else: thj = atan(dljy / dljx);

    if abs(thi - thj) <= pi_2:
        thr = (li * thi + lj * thj) / (li + lj);
    else:
        tmp = thj - pi * (thj / abs(thj));
        thr = li * thi + lj * tmp;
        thr /= (li + lj);

    cdef float sin_thr = sin(thr)
    cdef float cos_thr = cos(thr)

    cdef float axg = (ay - yg) * sin_thr + (ax - xg) * cos_thr;
    cdef float bxg = (by - yg) * sin_thr + (bx - xg) * cos_thr;
    cdef float cxg = (cy - yg) * sin_thr + (cx - xg) * cos_thr;
    cdef float dxg = (dy - yg) * sin_thr + (dx - xg) * cos_thr;

    cdef float delta1xg = min(axg,min(bxg,min(cxg,dxg)));
    cdef float delta2xg = max(axg,max(bxg,max(cxg,dxg)));

    return  delta1xg * cos_thr + xg, \
            delta1xg * sin_thr + yg, \
            delta2xg * cos_thr + xg, \
            delta2xg * sin_thr + yg

cdef _out_of_range(x, y, width, height):
    return x < 0 or y < 0 or x >= width or y >= height

cdef _line_angle_difference(x, y):
    cdef double diff = y - x
    diff = abs(math.atan2(sin(diff), cos(diff)))
    if diff > pi_2:
        diff = pi - diff
    return diff

class LineFunctions:

    @staticmethod
    def get_parallel_lines(point_a, point_b, length, distance=2):
        return _get_parallel_lines(point_a[0], point_a[1], point_b[0], point_b[1], length, distance)

    @staticmethod
    def get_intersection(line_a_point_a, line_a_point_b, line_b_point_a, line_b_point_b):
        result = _get_intersection(line_a_point_a, line_a_point_b, line_b_point_a, line_b_point_b)
        if result[0] == -1:
            return None
        return result    

    @staticmethod
    def get_line_samples(point_a, point_b, image, num_points):
        samples = list()
        for point in np.linspace(point_b, point_a, num_points):
            if not _out_of_range(point[0], point[1], image.shape[1], image.shape[0]): 
                samples.append(image[int(point[1]), int(point[0])])
        return samples

    @staticmethod
    def line_contour_confidence(point_a, point_b, image, num_points):
        samples = LineFunctions.get_line_samples(point_a, point_b, image, num_points)        
        slope = np.diff(samples)
        uniformity = 1 if len(slope) == 0 else 1.0 / (1 + np.mean(slope / 255.0))

        return uniformity * np.mean(samples) / 255.0

    @staticmethod
    def line_angle(x0, y0, x1, y1):
        #return np.arctan2(y1 - y0, x1 - x0)
        return math.atan2(float(y1 - y0), float(x1 - x0))

    @staticmethod #minimum angle between lines segments cannot differ by more than 90 degrees
    def line_angle_difference(x, y):
        return _line_angle_difference(x, y)

    @staticmethod
    def merge_line_pair(ax, ay, bx, by, cx, cy, dx, dy, dljx, dljy):
        return _merge_line_pair(ax, ay, bx, by, cx, cy, dx, dy, dljx, dljy)
