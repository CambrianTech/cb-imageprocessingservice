import math
import numpy as np
import time
import cv2
from cambrian.SegmentationLabel import SegmentationLabel
import cambrian.image_processing as ip

class RectangularSurface:
    def __init__(self, rf, model, votes, debug=None):
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
    def __init__(self, datum, vanishing_points, debug=None):
        self.datum = datum
        self.vanishing_points = vanishing_points
        self.debug = debug
        self.diagonal = math.hypot(self.datum['segmented'].shape[0], self.datum['segmented'].shape[1])
        self.find_contours()

    def find_contours(self, distance=0.9):

        shape = self.datum['segmented'].shape
        scale = 400 / self.diagonal
        ds_mask = cv2.resize(self.datum['segmented'], (int(shape[1] * scale), int(shape[0] * scale)), cv2.INTER_NEAREST)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13))

        def get_label_contours(label):
            isolated = np.zeros(ds_mask.shape, dtype=np.uint8)
            isolated[ds_mask == label] = 255
            isolated = cv2.dilate(isolated, kernel)
            isolated = cv2.resize(isolated, (shape[1], shape[0]))
            isolated[isolated<127] = 0
            contours, _ = cv2.findContours(isolated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            if self.debug is not None:
                self.debug = ip.overlay_mask(self.debug, isolated, hue=label*20)
                for cnt in contours:
                    cv2.drawContours(self.debug, [cnt], 0, (0,255,0), 3)
            
            return contours

        self.wall_contours = get_label_contours(SegmentationLabel.WALL)
               

            #isolated = cv2.dilate(isolated, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
            
                
                

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

        num_pts = len(self.vanishing_points)

        if num_pts < 2: return []

        first_index_space = self.vanishing_points[:num_pts // 2]
        second_index_space = self.vanishing_points[:num_pts]

        rectangles = []

        t = time.time()
        
        for ransac_iter in range(num_ransac_iter):
            if time.time() - t > max_time:
                break

            vp1 = np.random.choice(first_index_space)
            vp2 = np.random.choice(second_index_space)

            #print(vp1, vp2)

        return rectangles

