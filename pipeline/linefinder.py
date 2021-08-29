import cv2
import numpy as np

from pipeline.logging import log_image, im_logging_enabled, LogLevel

class Line():
    def __init__(self, data):
        self.data = data

    @property
    def a(self):
        return (self.data[0], self.data[1])

    @property
    def b(self):
        return (self.data[2], self.data[3])

    def draw(self, img, color=(255,0,0,255), thickness=1):
        cv2.line(img, self.a, self.b, color, thickness)


class LineFinder():

    def __init__(self, img, bw):
        super().__init__()
        self.img = img
        self.bw = bw

    def detect(self, data):
        self.height, self.width = self.img.shape[:2]
        self.diagonal = np.hypot(self.width, self.height)

        fld = cv2.ximgproc.createFastLineDetector(int(self.diagonal / 60.0), 1.41, 200, 240, 3, False)

        lines = map(lambda x: Line(x[0]), fld.detect(self.bw))

        if im_logging_enabled(data, LogLevel.Lines):
            debug = self.img.copy()
            [line.draw(debug) for line in lines]
            log_image(data, "lines", debug)

        return lines
