import cv2
import numpy as np
from scipy.spatial import distance
from collections.abc import Sequence

import math
from cambrian.LineFunctions import LineFunctions

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.logging import log_image, im_logging_enabled, LogLevel

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

    @property
    def point_a(self):
        return (self.data[0], self.data[1])

    @property
    def point_b(self):
        return (self.data[2], self.data[3])

    def draw(self, img, color=(180,255,100,255), thickness=1):
        cv2.line(img, self.point_a, self.point_b, color, thickness)

    def reshape(self, *args):
        return self.data.reshape(*args)

    def recalculate(self):
        self.midpoint = ((self.point_a[0] + self.point_b[0]) / 2, (self.point_a[1] + self.point_b[1]) / 2)
        self.length_sq = distance.sqeuclidean(self.point_a, self.point_b)
        self.length = math.sqrt(self.length_sq)
        self.angle = LineFunctions.line_angle(self.point_a[0], self.point_a[1], self.point_b[0], self.point_b[1])

    def bounding_box(self, width=10, length_multiplier=1.0, length_offset=0.0):
        return (self.midpoint, (max(self.length * length_multiplier, self.length + length_offset), width), np.degrees(self.angle))

class PipelineLineFinder(PipelineStep):

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.FindLines

    @property
    def required_keys(self) -> list:
        return ["image"]

    @property
    def output_keys(self) -> list:
        return ["lines"]

    #todo: write in C or lambda
    def merge(self, lines, search_width=5, search_length=1.1, angle_threshold=math.radians(3)):

        i=0
        while i < len(lines):
            line_a = lines[i]
            i += 1

            if line_a.dead: continue

            rect_a = line_a.bounding_box(search_width, length_multiplier=search_length)
            data = (line_a.point_a, line_a.point_b)

            parallel_lines = []
            max_length = line_a.length

            for j in range(i, len(lines)):
                line_b = lines[j]

                if line_b.dead: continue

                angle_diff = LineFunctions.line_angle_difference(line_a.angle, line_b.angle)

                if angle_diff <= angle_threshold:
                    rect_b = line_b.bounding_box(2, length_multiplier=search_length)
                    result, intersections = cv2.rotatedRectangleIntersection(rect_a, rect_b)

                    if not intersections is None:
                        parallel_lines.append(line_b)
                        data = LineFunctions.merge_lines(data, (line_b.point_a, line_b.point_b))

            if len(parallel_lines) > 0:
                parallel_lines.append(line_a)
                source_lines = parallel_lines
                new_line = Line(np.array([(data[0][0], data[0][1], data[1][0], data[1][1])], dtype=np.int).reshape(4))

                for line in parallel_lines:
                    line.dead = True

                lines.append(new_line)

        return list(filter(lambda x: not x.dead, lines))


    def run(self, data):

        bw = cv2.cvtColor(data["image"], cv2.COLOR_BGR2GRAY)

        self.height, self.width = bw.shape[:2]
        self.diagonal = np.hypot(self.width, self.height)

        fld = cv2.ximgproc.createFastLineDetector(int(self.diagonal / 60.0), 1.41, 200, 240, 3, False)

        sx = data["image"].shape[1] / data["hed"].shape[1]
        sy = data["image"].shape[0] / data["hed"].shape[0]
        lines = list(map(lambda x: Line(x.reshape(4)), fld.detect(bw)))

        lines.extend(list(map(lambda x: Line(x.reshape(4), sx, sy), fld.detect(data["hed"]))))

        if im_logging_enabled(data, LogLevel.Lines):
            debug = data["image"].copy()
            [line.draw(debug) for line in lines]
            log_image(data, "raw_lines", debug)

        lines = self.merge(lines, search_width=self.diagonal/200)

        if im_logging_enabled(data, LogLevel.Lines):
            debug = data["image"].copy()
            [line.draw(debug) for line in lines]
            log_image(data, "lines", debug)

        data["lines"] = lines
