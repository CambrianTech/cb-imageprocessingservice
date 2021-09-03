import cv2
import numpy as np
import math
from scipy.spatial import distance
from collections.abc import Sequence
from bisect import bisect_left, bisect_right

from cambrian.LineFunctions import LineFunctions

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.logging import log_image, im_logging_enabled, LogLevel

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

def sharpen(img, alpha=1.5, beta=-1.0, kernel_size = 21):
    smoothed = cv2.GaussianBlur(img, (kernel_size, kernel_size), kernel_size)
    return cv2.addWeighted(img, alpha, smoothed, beta, 0)

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

    def bounding_box(self, width, length_multiplier=1.0):
        return (self.midpoint, (self.length * length_multiplier, width), np.degrees(self.angle))

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
    def merge(self, lines, search_width, search_length=1.1, angle_threshold=math.radians(3)):

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

            start_index = bisect_right(angles, line_a.angle - angle_threshold, 0, i)
            stop_index = bisect_left(angles, line_a.angle + angle_threshold, i)

            for line_b in lines[start_index:stop_index]:

                #Optimization possible: line_angle_difference should not be required by bisect methods above returning only angles in range
                if line_a == line_b or line_b.dead or LineFunctions.line_angle_difference(line_a.angle, line_b.angle) > angle_threshold: continue

                dist_sq = distance.sqeuclidean(line_a.midpoint, line_b.midpoint)

                if dist_sq <= min_dist_sq:
                    result = 1
                else:
                    rect_b = line_b.bounding_box(search_width, length_multiplier=search_length)
                    result, _ = cv2.rotatedRectangleIntersection(rect_a, rect_b)

                if result != 0:
                    line_a.dead = True
                    line_b.dead = True
                    data = LineFunctions.merge_lines(data, (line_b.point_a, line_b.point_b))


            if line_a.dead:
                new_line = Line(np.array([(data[0][0], data[0][1], data[1][0], data[1][1])], dtype=np.int).reshape(4))
                insert = bisect_left(angles, new_line.angle)
                lines.insert(insert, new_line)
                angles.insert(insert, new_line.angle)
                i = min(insert, i)

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

        fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 60.0), 1.41, 200, 240, 3, False)

        lines = list(map(lambda x: Line(x.reshape(4)), fld.detect(bw)))
        log_lines(lines, "bw_lines")

        sx = data["image"].shape[1] / data["hed"].shape[1]
        sy = data["image"].shape[0] / data["hed"].shape[0]

        hed = data["hed"]
        hed_lines = list(map(lambda x: Line(x.reshape(4), sx, sy), fld.detect(hed)))
        before = len(hed_lines)
        hed_lines = self.merge(hed_lines, search_width=diagonal/100)
        #print("before: %d, after: %d" % (before, len(hed_lines)))

        log_lines(hed_lines, "hed_lines")
        lines.extend(hed_lines)

        normals = np.uint8(data["normals"])
        sx = data["image"].shape[1] / normals.shape[1]
        sy = data["image"].shape[0] / normals.shape[0]
        normals = cv2.split(normals)
        normals_lines = []
        for i in range(0, 3):
            normals_lines.extend(list(map(lambda x: Line(x.reshape(4), sx, sy), fld.detect(normals[i]))))
        normals_lines = self.merge(normals_lines, search_width=diagonal/100)
        log_lines(normals_lines, "normals_lines")
        lines.extend(normals_lines)

        lines = self.merge(lines, search_width=diagonal/200)

        log_lines(lines, "merged_lines")

        data["lines"] = lines
