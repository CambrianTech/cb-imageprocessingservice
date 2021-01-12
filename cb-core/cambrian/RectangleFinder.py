import math
import numpy as np
import time
import cv2
from cambrian.SegmentationLabel import SegmentationLabel, SegmentationSet
from cambrian.LineFunctions import LineFunctions
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
        self.line_sets = None

    @property
    def label(self):
        return SegmentationSet.label(self.label_set)

class RectangleFinder:
    def __init__(self, datum, vanishing_points, label_sets=[SegmentationSet.FLOOR, SegmentationSet.CEILING, SegmentationSet.WALL], debug=None):
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

        def get_label_mask(label_data):
            isolated = np.zeros(ds_mask.shape, dtype=np.uint8)
            for label in label_set:
                isolated[ds_mask == label] = 255
            isolated = cv2.erode(isolated, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
            isolated = cv2.dilate(isolated, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(9,9)))
            isolated[isolated<127] = 0
            label_data.mask = isolated
            return label_data
        
        #tues april 13 2:30
        def get_line_sets(label_data):
            label_data.line_sets = []

            for vp in self.vanishing_points:
                matches = []
                for line in vp.inliers:
                    if line.label in label_data.label_set: matches.append(line)
                    else:
                        point_a, point_b = line.translated_points(label_data.mask.shape[1], label_data.mask.shape[0])
                        samples = LineFunctions.get_line_samples(point_a, point_b, label_data.mask, 5)

                        #check within expanded mask and decent probability of label type (could check all in set)
                        if len(samples) > 0 and max(samples) > 0 and line.get_probability(label_set[0]) > 0.1: 
                            matches.append(line)

                if len(matches):
                    label_data.line_sets.append(matches)
            return label_data

        self.data = {}

        for label_set in self.label_sets:
            value = LabelData(label_set)
            get_label_mask(value)
            get_line_sets(value)
            self.data[value.label] = value

        if self.debug is not None:
            for key in self.data:
                datum = self.data[key]
                color = SegmentationSet.color(datum.label_set)
                hue = ip.convert_color(color, cv2.COLOR_BGR2HSV_FULL)[0]
                mask = cv2.resize(datum.mask, (shape[1], shape[0]))
                self.debug = ip.overlay_mask(self.debug, mask, hue=hue)

            for key in self.data:
                datum = self.data[key]

                if key != SegmentationLabel.WALL: continue

                color = SegmentationSet.color(datum.label_set)
                for lines in datum.line_sets:
                    for line in lines:
                        line.draw(self.debug, color=color, thickness=3)
            
    
                

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

