import cv2
import numpy as np
import math
from scipy.spatial import distance
from collections.abc import Sequence
from bisect import bisect_left, bisect_right

from cambrian.LineFunctions import LineFunctions
from cambrian.frei_chen import frei_chen

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.logging import log_image, im_logging_enabled, LogLevel
from pipeline.Line import Line, line_angle_difference, bounding_box

def point_segment_distance(px, py, x1, y1, x2, y2):
  dx = x2 - x1
  dy = y2 - y1
  if dx == dy == 0:  # the segment's just a point
    return math.hypot(px - x1, py - y1)

  # Calculate the t that minimizes the distance.
  t = ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)

  # See if this represents one of the segment's
  # end points or a point in the middle.
  if t < 0:
    dx = px - x1
    dy = py - y1
  elif t > 1:
    dx = px - x2
    dy = py - y2
  else:
    near_x = x1 + t * dx
    near_y = y1 + t * dy
    dx = px - near_x
    dy = py - near_y

  return math.hypot(dx, dy)

def gabor(bw, theta, lambd, gamma = 0.0, psi = 0.0):
    ksize = lambd
    sigma = ksize * lambd
    result = cv2.filter2D(bw, cv2.CV_8UC1, cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F))
    return result

def sharpen(img, alpha=1.5, beta=-1.0, kernel_size = 21):
    smoothed = cv2.GaussianBlur(img, (kernel_size, kernel_size), kernel_size)
    return cv2.addWeighted(img, alpha, smoothed, beta, 0)

class PipelineLineFinder(PipelineStep):

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.FindLines

    @property
    def required_keys(self) -> list:
        return ["image", "hed", "normals"]

    @property
    def output_keys(self) -> list:
        return ["lines"]


    #todo: write in C or lambda
    def merge(self, lines, search_width, search_length=1.01, angle_threshold=math.radians(3)):

        min_dist_sq = search_width * search_width

        for i in range(len(lines)):
            line_a = lines[i]
            
            if line_a.dead: continue

            rect_a = bounding_box(line_a, search_width, length_multiplier=search_length)
            data = (line_a.point_a, line_a.point_b)

            for j in range(len(lines)):
                line_b = lines[j]
                #Optimization possible: line_angle_difference should not be required by bisect methods above returning only angles in range
                if i == j or line_b.dead or line_angle_difference(line_a.angle, line_b.angle) > angle_threshold: continue

                dist_sq = distance.sqeuclidean(line_a.midpoint, line_b.midpoint)

                if dist_sq <= min_dist_sq:
                    result = 1
                else:
                    rect_b = bounding_box(line_b, search_width, length_multiplier=search_length)
                    result, _ = cv2.rotatedRectangleIntersection(rect_a, rect_b)

                if result != 0:
                    line_a.dead = True
                    line_b.dead = True
                    data = LineFunctions.merge_lines(data, (line_b.point_a, line_b.point_b))


            if line_a.dead:
                new_line = Line(data[0][0], data[0][1], data[1][0], data[1][1])
                lines.insert(i, new_line)

        return list(filter(lambda x: not x.dead, lines))

    
    def run(self, data):

        bw = cv2.cvtColor(data["image"], cv2.COLOR_RGB2GRAY)

        self.height, self.width = bw.shape[:2]
        diagonal = np.hypot(self.width, self.height)

        def log_lines(lines, name):
            if im_logging_enabled(data, LogLevel.Lines):
                debug = data["image"].copy()
                thickness = max(int(math.hypot(debug.shape[0], debug.shape[1]) / 600), 1)
                [line.draw(debug,  thickness=thickness) for line in lines]
                log_image(data, name, debug)

        #transform_result = lambda x: list(map(lambda x: Line(x.reshape(4), sx, sy), result))
        transform_result = lambda result: list(map(lambda x: Line(x[0][0] * sx, x[0][1] * sy, x[0][2] * sx, x[0][3] * sy), result))

        fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 60.0), 1.41, 200, 240, 3, False)
        lines = []

        #find lines in BW image
        sx = data["image"].shape[1] / bw.shape[1]
        sy = data["image"].shape[0] / bw.shape[0]
        result = fld.detect(bw)
        if result is not None and len(result) > 0: 
            lines.extend(transform_result(result))
        log_lines(lines, "bw_lines")


        #find lines in hed hed edges
        sx = data["image"].shape[1] / data["hed"].shape[1]
        sy = data["image"].shape[0] / data["hed"].shape[0]

        result = fld.detect(data["hed"])
        if result is not None and len(result) > 0: 
            #lines are made parallel by thickness of source image edges
            hed_lines = self.merge(transform_result(result), search_width=diagonal/100)
            log_lines(hed_lines, "hed_lines")
            lines.extend(hed_lines)

        #find lines in normals
        fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 20.0), 1.41, 200, 240, 3, False)
        normals = np.uint8(data["normals"])
        #log_image(data, "normals", normals)
        sx = data["image"].shape[1] / normals.shape[1]
        sy = data["image"].shape[0] / normals.shape[0]
        normals = cv2.split(normals)
        normals_lines = []
        for i in range(0, 3):
            result = fld.detect(normals[i])
            if result is not None and len(result) > 0:
                normals_lines.extend(transform_result(result))
        
        if len(normals_lines) > 0:
            #cleanup normals
            normals_lines = self.merge(normals_lines, search_width=diagonal/300)
            log_lines(normals_lines, "normals_lines")
            lines.extend(normals_lines)

        #find lines in gabor edges:
        gabor_scale = 1500.0 / diagonal
        bw_res = cv2.resize(bw, (int(self.width * gabor_scale), int(self.height * gabor_scale)), cv2.INTER_CUBIC) if gabor_scale < 1.0 else bw
        v_gabor = gabor(bw_res, 0, 7)
        h_gabor = gabor(bw_res, np.pi/2.0, 9)
        edges = cv2.addWeighted(v_gabor, 3.0, h_gabor, 3.0, -20)
        edges = cv2.bilateralFilter(edges, 5, 5, 5)
        edges = cv2.resize(edges, (self.width, self.height), interpolation = cv2.INTER_CUBIC)

        sx = data["image"].shape[1] / edges.shape[1]
        sy = data["image"].shape[0] / edges.shape[0]
        result = fld.detect(edges)
        if result is not None and len(result) > 0: 
            gabor_lines = self.merge(transform_result(result), search_width=diagonal/100)
            log_lines(gabor_lines, "gabor_lines")
            lines.extend(gabor_lines)

        #frei chen edges:
        clean_edges = frei_chen(bw)
        result = fld.detect(bw - (clean_edges * 5.0).astype("uint8"))
        if result is not None and len(result) > 0: 
            frei_lines = transform_result(result)
            log_lines(frei_lines, "frei_lines")
            lines.extend(frei_lines)


        #merge all
        lines = self.merge(lines, search_width=diagonal/200)

        log_lines(lines, "merged_lines")

        data["lines"] = lines
