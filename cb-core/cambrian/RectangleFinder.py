import math
import numpy as np
import time

class RectangularSurface:
    def __init__(self, rf, model, votes):
        self.vpf = vpf
        self.votes = votes
        self._score = None

    @property
    def score(self):
        if self._score is None:
            self._score = sum(self.votes)
        return self._score

    def clear_indexes(self, indexes):
        self.votes[indexes] = 0
        self._score = None

class RectangleFinder:
    def __init__(self, images, vanishing_points):
        self.images = images
        self.vanishing_points = vanishing_points

    def compute(self, num_ransac_iter=2000, threshold_inlier=math.radians(5), max_time=1.0):
        """Estimate rectangular surfaces using Ransac.
        Parameters
        ----------
        num_ransac_iter: int
            Number of iterations to run ransac.
        threshold_inlier: float
            threshold to be used for computing inliers in radians.
        max_points: int
            max number of vanishing points to return
        Returns
        -------
        rectangles: list
            list of RectangularSurface objects, sorted by score.
        """

        rectangles = []

        t = time.time()
        
        for ransac_iter in range(num_ransac_iter):
            if time.time() - t > max_time:
                break

        return rectangles

