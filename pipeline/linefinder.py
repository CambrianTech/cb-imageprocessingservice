import cv2
import numpy as np
import math
from scipy.spatial import distance
from collections.abc import Sequence
from bisect import bisect_left

from cambrian.LineFunctions import LineFunctions

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.logging import log_image, im_logging_enabled, LogLevel

#a[pos].angle == x.angle

def binary_search(lines, x, angle_threshold, lo=0, hi=None):
    if hi is None: hi = len(lines)
    pos = bisect_left(lines, x, lo, hi)                  # find insertion position
    return pos if pos != hi and LineFunctions.line_angle_difference(lines[pos].angle, x.angle) < angle_threshold else -1  # don't walk off the end

class Line(Sequence):
    def __init__(self, data, sx=1, sy=1):
        super().__init__()
        self.data = data
        self.data[0] *= sx
        self.data[2] *= sx
        self.data[1] *= sy
        self.data[3] *= sy
        self.recalculate()

        self.dead = False

    def __getitem__(self, i):
        return self.data[i]

    def __len__(self):
        return len(self.data)

    def __lt__(self, other):
        return self.angle < other.angle
    
    def __eq__(self, other):
        return self.angle == other.angle

    @property
    def point_a(self):
        return (self.data[0], self.data[1])

    @property
    def point_b(self):
        return (self.data[2], self.data[3])

    def draw(self, img, color=(255,50,255,255), thickness=2):
        cv2.line(img, self.point_a, self.point_b, color, thickness)

    def reshape(self, *args):
        return self.data.reshape(*args)

    def recalculate(self):
        self.midpoint = ((self.point_a[0] + self.point_b[0]) / 2, (self.point_a[1] + self.point_b[1]) / 2)
        self.length_sq = distance.sqeuclidean(self.point_a, self.point_b)
        self.length = math.sqrt(self.length_sq)
        self.angle = LineFunctions.line_angle(self.point_a[0], self.point_a[1], self.point_b[0], self.point_b[1])

    def bounding_box(self, width=10, length_multiplier=1.0):
        return (self.midpoint, (max(self.length * length_multiplier, self.length), width), np.degrees(self.angle))

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
    def merge(self, lines, search_width, search_length=1.1, angle_threshold=math.radians(4)):

        lines = sorted(lines)
        angles = list(map(lambda x: x.angle, lines))

        min_dist_sq = search_width * search_width

        i=0
        while i < len(lines):
            line_a = lines[i]
            i += 1

            if line_a.dead: continue

            rect_a = line_a.bounding_box(search_width, length_multiplier=search_length)
            data = (line_a.point_a, line_a.point_b)

            parallel_lines = []
            max_length = line_a.length

            stop_index = bisect_left(angles, line_a.angle + angle_threshold, i)

            for line_b in lines[i:stop_index]:

                if line_b.dead: continue

                dist_sq = distance.sqeuclidean(line_a.midpoint, line_b.midpoint)

                if dist_sq <= min_dist_sq:
                    result = 1
                else:
                    rect_b = line_b.bounding_box(search_width, length_multiplier=search_length)
                    result, _ = cv2.rotatedRectangleIntersection(rect_a, rect_b)

                # rect_b = line_b.bounding_box(search_width, length_multiplier=search_length)
                # result, _ = cv2.rotatedRectangleIntersection(rect_a, rect_b)

                if result != 0:
                    parallel_lines.append(line_b)
                    data = LineFunctions.merge_lines(data, (line_b.point_a, line_b.point_b))


            if len(parallel_lines) > 0:
                parallel_lines.append(line_a)
                source_lines = parallel_lines
                new_line = Line(np.array([(data[0][0], data[0][1], data[1][0], data[1][1])], dtype=np.int).reshape(4))

                for line in parallel_lines:
                    line.dead = True

                insert = bisect_left(angles, new_line.angle, i)
                lines.insert(insert, new_line)
                angles.insert(insert, new_line.angle)

        return list(filter(lambda x: not x.dead, lines))

    
    def run(self, data):

        bw = cv2.cvtColor(data["image"], cv2.COLOR_RGB2GRAY)

        self.height, self.width = bw.shape[:2]
        diagonal = np.hypot(self.width, self.height)

        def log_lines(lines, name):
            if im_logging_enabled(data, LogLevel.Lines):
                debug = data["image"].copy()
                [line.draw(debug) for line in lines]
                log_image(data, name, debug)

        fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 60.0), 1.41, 200, 240, 3, False)

        lines = list(map(lambda x: Line(x.reshape(4)), fld.detect(bw)))
        log_lines(lines, "bw_lines")

        sx = data["image"].shape[1] / data["hed"].shape[1]
        sy = data["image"].shape[0] / data["hed"].shape[0]
        hed_lines = list(map(lambda x: Line(x.reshape(4), sx, sy), fld.detect(data["hed"])))
        log_lines(hed_lines, "hed_lines")
        lines.extend(hed_lines)

        normals = np.uint8(data["normals"])
        sx = data["image"].shape[1] / normals.shape[1]
        sy = data["image"].shape[0] / normals.shape[0]
        normals = cv2.split(normals)
        normals_lines = []
        for i in range(0, 3):
            normals_lines.extend(list(map(lambda x: Line(x.reshape(4), sx, sy), fld.detect(normals[i]))))
        log_lines(normals_lines, "normals_lines")
        lines.extend(normals_lines)

        lines = self.merge(lines, search_width=diagonal/180)

        log_lines(lines, "merged_lines")

        data["lines"] = lines
