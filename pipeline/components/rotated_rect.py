import math
import numpy as np
import numba as nb
import cv2

from .line import Line
from .point import Point

class RotatedRect(tuple):

    def __new__(cls, x):
        return tuple.__new__(cls, x)

    @property
    def center(self) -> Point:
        return Point(self[0])

    @property
    def width(self) -> float:
        return self[1][0]

    @property
    def height(self) -> float:
        return self[1][1]

    @property
    def length(self) -> float:
        return max(self.width, self.height)

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def empty(self) -> bool:
        return max(self.width, self.height) == 0

    @property
    def angle(self) -> float:
        return self.line.angle

    @property
    def points(self):
        if self._points is None:
            self._points = np.int0(cv2.boxPoints(self))
        return self._points

    _points = None
    @property
    def points(self):
        if self._points is None:
            self._points = np.int0(cv2.boxPoints(self))
        return self._points

    def intersects(self, other):
        result, _ = cv2.rotatedRectangleIntersection(self, other)
        return result != 0

    def get_intersection(self, other):
        result, vertices = cv2.rotatedRectangleIntersection(self, other)
        
        if vertices is None:
            return None

        return np.mean(vertices, axis=(0,1))

    _line = None
    @property
    def line(self):
        if self._line is None:
            if self.width > self.height:
                point_a = (self.points[0][0] + self.points[1][0]) / 2, (self.points[0][1] + self.points[1][1]) / 2
                point_b = (self.points[2][0] + self.points[3][0]) / 2, (self.points[2][1] + self.points[3][1]) / 2 
            else:
                point_a = (self.points[1][0] + self.points[2][0]) / 2, (self.points[1][1] + self.points[2][1]) / 2
                point_b = (self.points[3][0] + self.points[0][0]) / 2, (self.points[3][1] + self.points[0][1]) / 2 

            self._line = Line(point_a[0], point_a[1], point_b[0], point_b[1])
        return self._line

    def resized(self, length_factor=1.0, length_offset=0.0, width_factor=1.0, width_offset=0.0):
        extended_size = (length_factor * self[1][0] + length_offset, width_factor * self[1][1] + width_offset) if self[1][0] > self[1][1] else (width_factor * self[1][0] + width_offset, length_factor * self[1][1] + length_offset)
        return RotatedRect((self[0], extended_size, self[2]))
