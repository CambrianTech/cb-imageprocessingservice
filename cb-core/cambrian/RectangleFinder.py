import math
import numpy as np
import time
import cv2
from cambrian.SegmentationLabel import SegmentationLabel, SegmentationSet
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

class LabelData:
    def __init__(self, label_set):
        self.label_set = label_set
        self.mask = None

class RectangleFinder:
    def __init__(self, datum, vanishing_points, label_sets=[SegmentationSet.WALL, SegmentationSet.FLOOR, SegmentationSet.CEILING], debug=None):
        self.datum = datum
        self.vanishing_points = vanishing_points
        self.debug = debug
        self.diagonal = math.hypot(self.datum['segmented'].shape[0], self.datum['segmented'].shape[1])
        self.label_sets = label_sets
        self.initialize_data()


    def initialize_data(self, distance=0.9):

        shape = self.datum['segmented'].shape
        scale = 200 / self.diagonal
        ds_mask = cv2.resize(self.datum['segmented'], (int(shape[1] * scale), int(shape[0] * scale)), cv2.INTER_NEAREST)

        def get_label_mask(label_set):
            isolated = np.zeros(ds_mask.shape, dtype=np.uint8)
            for label in label_set:
                isolated[ds_mask == label] = 255
            isolated = cv2.erode(isolated, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)))
            isolated = cv2.dilate(isolated, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(13,13)))
            isolated = cv2.resize(isolated, (shape[1], shape[0]))
            isolated[isolated<127] = 0

            if self.debug is not None:
                self.debug = ip.overlay_mask(self.debug, isolated, hue=label_set[0]*13)
            
            return isolated

        def get_line_sets(label_set):
            line_sets = []

            color = SegmentationSet.color(label_set)
            for vp in self.vanishing_points:
                line_set = []

                matches = list(filter(lambda x: x.label in label_set, vp.inliers))

                if len(matches):
                    if self.debug is not None:
                        for line in matches:
                            line.draw(self.debug, color=color, thickness=3)

                    line_sets.append(matches)

            return line_sets

        self.data = {}

        for label_set in self.label_sets:
            key = SegmentationSet.key(label_set)
            self.data[key] = LabelData(label_set)
            self.data[key].mask = get_label_mask(label_set)
            self.data[key].line_sets = get_line_sets(label_set)

        #exit()
        # self.line_sets = []
        # for vp in self.vanishing_points:
        #     self.line_sets.extend(self.get_line_sets(vp))

        # self.wall_mask = get_label_mask(SegmentationLabel.WALL)
        # self.floor_mask = get_label_mask(SegmentationLabel.FLOOR)
        # self.ceiling_mask = get_label_mask(SegmentationLabel.CEILING)       

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

