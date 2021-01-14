import math
import numpy as np
import time

class VanishingPoint:
    def __init__(self, vpf, model, votes):
        self.vpf = vpf
        self.model = model
        self.votes = votes
        
        self._score = sum(self.votes)
        if np.any(self.vpf.seeds != None):
            cm = self.model[:2] / self.model[2]
            dot = 1 - abs(np.dot(cm / np.linalg.norm(cm), self.vpf.seeds[:2] / np.linalg.norm(self.vpf.seeds[:2])))
            self._score = self._score * dot > .9

    def __eq__(self, other):
        return self.score() == other.score()

    def __lt__(self, other):
        return self.score() < other.score()

    @property
    def score(self):
        return self._score
        

class VanishingPointFinder:
    def __init__(self, line_data, seeds=None):
        self.line_data = line_data
        self.edgelets = self.compute_edgelets()
        self.seeds = seeds

    def compute_votes(self, model, threshold_inlier):
        """Compute votes for each of the edgelet against a given vanishing point.
        Votes for edgelets which lie inside threshold are same as their strengths,
        otherwise zero.
        Parameters
        ----------
        edgelets: tuple of ndarrays
            (locations, directions, strengths) as computed by `compute_edgelets`.
        model: ndarray of shape (3,)
            Vanishing point model in homogenous cordinate system.
        threshold_inlier: float
            Threshold to be used for computing inliers in degrees. Angle between
            edgelet direction and line connecting the  Vanishing point model and
            edgelet location is used to threshold.
        Returns
        -------
        votes: ndarry of shape (n_edgelets,)
            Votes towards vanishing point model for each of the edgelet.
        """
        vp = model[:2] / model[2]

        locations, directions, strengths = self.edgelets[:3]

        est_directions = locations - vp

        dot_prod = np.sum(est_directions * directions, axis=1)
        abs_prod = np.linalg.norm(directions, axis=1) * \
                   np.linalg.norm(est_directions, axis=1)
        abs_prod[abs_prod == 0] = 1e-5

        cosine_theta = np.abs(dot_prod / abs_prod)

        theta_thresh = np.cos(threshold_inlier)

        return (cosine_theta > theta_thresh) * strengths

    def compute_edgelets(self, class_labels=None):
        locations = []
        directions = []
        strengths = []
        classes = []

        for line in self.line_data:
            p0, p1 = np.array([line.point_a[0], line.point_a[1]]), np.array([line.point_b[0], line.point_b[1]])
            if class_labels is not None:
                classes.append(class_labels[int(line.midpoint[1]), int(line.midpoint[0])])

            locations.append(line.midpoint)
            directions.append(p1 - p0)
            strengths.append(line.length)

        locations = np.array(locations)
        directions = np.array(directions)
        strengths = np.array(strengths)
        classes = np.array(classes)

        directions = np.array(directions) / np.linalg.norm(directions, axis=1)[:, np.newaxis]

        return (locations, directions, strengths, classes)

    def edgelet_lines(self, edgelets):
        """Compute lines in homogenous system for edglets.
        Parameters
        ----------
        edgelets: tuple of ndarrays
            (locations, directions, strengths) as computed by `compute_edgelets`.
        Returns
        -------
        lines: ndarray of shape (n_edgelets, 3)
            Lines at each of edgelet locations in homogenous system.
        """
        locations, directions = edgelets[:2]
        normals = np.zeros_like(directions)
        normals[:, 0] = directions[:, 1]
        normals[:, 1] = -directions[:, 0]
        p = -np.sum(locations * normals, axis=1)
        lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
        return lines

    def compute(self, num_ransac_iter=2000, threshold_inlier=math.radians(5), max_time=1.0, find_vert=True):
        """Estimate vanishing point using Ransac.
        Parameters
        ----------
        edgelets: tuple of ndarrays
            (locations, directions, strengths) as computed by `compute_edgelets`.
        num_ransac_iter: int
            Number of iterations to run ransac.
        threshold_inlier: float
            threshold to be used for computing inliers in degrees.
        Returns
        -------
        best_model: ndarry of shape (3,)
            Best model for vanishing point estimated.
        Reference
        ---------
        Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
        "Auto-rectification of user photos." 2014 IEEE International Conference on
        Image Processing (ICIP). IEEE, 2014.
        """
        locations, directions, strengths = self.edgelets[:3]
        lines = self.edgelet_lines(self.edgelets)

        num_pts = strengths.size

        arg_sort = np.argsort(-strengths)
        first_index_space = arg_sort[:num_pts // 5]
        second_index_space = arg_sort[:num_pts // 2]

        best_model = None
        vanishing_points = []
        t = time.time()

        for ransac_iter in range(num_ransac_iter):
            if time.time() - t > max_time:
                return best_model

            ind1 = np.random.choice(first_index_space)

            # if classes[ind1]!=0 and classes[ind1]!=3: continue

            ind2 = np.random.choice(second_index_space)

            l1 = lines[ind1]
            l2 = lines[ind2]

            current_model = np.cross(l1, l2)

            if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
                # reject degenerate candidates
                continue


            if find_vert:
                if current_model[1] / current_model[2] < 1000: continue

                dt1 = abs(np.dot(directions[ind1], [0, 1]))
                dt2 = abs(np.dot(directions[ind2], [0, 1]))

                if dt1 < .95 or dt2 < .95:
                    continue

            current_model = current_model / current_model[2]

            vp = VanishingPoint(self, current_model, self.compute_votes(current_model, threshold_inlier))
            
            vanishing_points.append(vp)

        vanishing_points.sort(key=lambda x:x.score, reverse=True)

        return vanishing_points
        
