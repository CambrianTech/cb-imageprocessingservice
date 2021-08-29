import cv2
import numpy as np
from scipy.spatial import distance
from collections.abc import Sequence

import math
from cambrian.LineFunctions import LineFunctions
from pipeline.logging import log_image, im_logging_enabled, LogLevel

class Line(Sequence):
    def __init__(self, data):
        super().__init__()
        self.data = data
        self.recalculate()

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

    def draw(self, img, color=(255,0,0,255), thickness=1):
        cv2.line(img, self.point_a, self.point_b, color, thickness)


    def recalculate(self):
        self.midpoint = ((self.point_a[0] + self.point_b[0]) / 2, (self.point_a[1] + self.point_b[1]) / 2)
        self.length_sq = distance.sqeuclidean(self.point_a, self.point_b)
        self.length = math.sqrt(self.length_sq)
        self.angle = LineFunctions.line_angle(self.point_a[0], self.point_a[1], self.point_b[0], self.point_b[1])

class LineFinder():

    def __init__(self, img, bw):
        super().__init__()
        self.img = img
        self.bw = bw

    def detect(self, data):
        self.height, self.width = self.img.shape[:2]
        self.diagonal = np.hypot(self.width, self.height)

        fld = cv2.ximgproc.createFastLineDetector(int(self.diagonal / 60.0), 1.41, 200, 240, 3, False)

        lines = list(map(lambda x: Line(x.reshape(4)), fld.detect(self.bw)))

        if im_logging_enabled(data, LogLevel.Lines):
            debug = self.img.copy()
            [line.draw(debug) for line in lines]
            log_image(data, "lines", debug)

        return lines
