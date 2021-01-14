import math
import numpy as np
import time

class RectangleFinder:
    def __init__(self, line_data):
        self.line_data = line_data

    def compute(self, num_ransac_iter=2000, threshold_inlier=math.radians(5), max_time=1.0):
        print("Does nothing yet")
