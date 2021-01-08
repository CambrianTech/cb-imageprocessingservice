import math
import numpy as np
import time
from sklearn.cluster import SpectralClustering

class VanishingPoint:
    def __init__(self, vpf, model, votes):
        self.vpf = vpf
        self.point = (model / model[2])[:2]
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

        if len(self.line_data) < 5: return None

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

    def compute(self, num_ransac_iter=2000, threshold_inlier=math.radians(5), max_points=10, max_time=1.0):
        """Estimate vanishing point using Ransac.
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
        vanishing_points: list
            list of VanishingPoint objects, sorted by score.
        Reference
        ---------
        Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
        "Auto-rectification of user photos." 2014 IEEE International Conference on
        Image Processing (ICIP). IEEE, 2014.
        """
        if self.edgelets is None: return []

        locations, directions, strengths = self.edgelets[:3]
        lines = self.edgelet_lines(self.edgelets)

        num_pts = strengths.size

        arg_sort = np.argsort(-strengths)
        first_index_space = arg_sort[:num_pts // 5]
        second_index_space = arg_sort[:num_pts // 2]
        vanishing_points = []
        t = time.time()

        for ransac_iter in range(num_ransac_iter):
            if time.time() - t > max_time:
                break

            ind1 = np.random.choice(first_index_space)

            # if classes[ind1]!=0 and classes[ind1]!=3: continue

            ind2 = np.random.choice(second_index_space)

            l1 = lines[ind1]
            l2 = lines[ind2]

            current_model = np.cross(l1, l2)

            if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
                # reject degenerate candidates
                continue

            vp = VanishingPoint(self, current_model, self.compute_votes(current_model, threshold_inlier))
            
            vanishing_points.append(vp)

        filtered = []
        while len(vanishing_points) > 0 and len(filtered) < max_points:
            vanishing_points.sort(key=lambda x:x.score, reverse=True)
            vp = vanishing_points.pop(0)
            if vp.score == 0:
                break
            filtered.append(vp)
            to_remove = np.where(vp.votes > 0)
            for vp in vanishing_points:
                vp.clear_indexes(to_remove)

        return filtered

        # if k + 2 < len(filtered):
        #     X = []
        #     for vp in filtered:
        #         print(vp.score)
        #         X.append(vp.point)

        #     print(len(filtered), k)
        #     X = np.array(X)
        #     labels = SpectralClustering(n_clusters=k, assign_labels="kmeans", affinity='nearest_neighbors', random_state=0).fit_predict(X)
        #     labels = labels.tolist()

        #     best_points = []
        #     for i in range(k):
        #         index = labels.index(i)
        #         best_points.append(filtered[index])

        #     return best_points
        # else:
        #     return filtered
        
