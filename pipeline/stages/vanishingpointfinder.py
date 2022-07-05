from pickle import FALSE
from random import random
import numpy as np
from scipy import ndimage
import cv2
import math
import time
from enum import IntEnum
from scipy.spatial import distance

from cambrian.LineFunctions import LineFunctions
from sklearn import cluster
from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.surface_type import SurfaceType
from pipeline.misc.utils import resize_array, random_color, overlay_mask, partition
from .planegeometry import Dimension
from .extractsurfaces import box_like, legged_objects
from pipeline.data.logging import log_image, log_segmentation_image, im_logging_enabled
from pipeline.components.line import Line, line_on_image_edge, merge_lines, draw_lines
from pipeline.data.ade20k import ADE20K

find_horizontal = True

def angle_with_vp(model, locations, directions):

    vp = model[:2] / model[2]

    est_directions = locations - vp

    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
               np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = np.abs(dot_prod / abs_prod)

    return cosine_theta

def compute_votes(edgelets, model, threshold_inlier):

    cosine_theta = angle_with_vp(model, edgelets.locations, edgelets.directions)
    theta_thresh = np.cos(threshold_inlier)

    return (cosine_theta > theta_thresh) * edgelets.strengths

def get_votes(lines, model, angle_threshold=np.radians(3)):
    
    locations = np.array(list(map(lambda x: x.midpoint, lines)))
    directions = np.array(list(map(lambda x: x.direction, lines)))
    strengths = np.array(list(map(lambda x: x.length, lines)))
    #directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    cosine_thetas = angle_with_vp(model, locations, directions)

    theta_thresh = np.cos(angle_threshold)
    
    return (cosine_thetas > theta_thresh) * strengths

def get_angles(lines, model):
    
    locations = np.array(list(map(lambda x: x.midpoint, lines)))
    directions = np.array(list(map(lambda x: x.direction, lines)))
    strengths = np.array(list(map(lambda x: x.length, lines)))
    #directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    cosine_thetas = angle_with_vp(model, locations, directions)
    
    return cosine_thetas

def get_inliers(lines, model, angle_threshold=np.radians(3)):
    
    locations = np.array(list(map(lambda x: x.midpoint, lines)))
    directions = np.array(list(map(lambda x: x.direction, lines)))
    #directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]

    cosine_thetas = angle_with_vp(model, locations, directions)

    theta_thresh = np.cos(angle_threshold)
    indices = np.argwhere(cosine_thetas > theta_thresh).flatten()
    
    return [lines[i] for i in indices]

def is_inlier(line, model, angle_threshold=np.radians(3)):
    return len(get_inliers([line], model, angle_threshold)) > 0

def remove_inliers(lines, model, angle_threshold=np.radians(3)):
    inliers = get_inliers(lines, model, angle_threshold)
    return [line for line in lines if line not in inliers]

def compute_edgelets(lines):

    if len(lines) < 2: return None

    locations = []
    directions = []
    strengths = []

    for line in lines:

        locations.append(line.midpoint)
        directions.append(line.direction)
        strengths.append(line.length)

    locations = np.array(locations)
    strengths = np.array(strengths)
    directions = np.array(directions)

    return Edgelets(locations, directions, strengths)

def get_vanishing_point_lines(lines, model, extend=1.0):

    for i in range(len(lines)):
        line = lines[i]
        isdead = line.dead
        midpt = line.midpoint
        dir  = model[:2]/model[2] - midpt
        dir /= np.linalg.norm(dir)
        pt1 = midpt + dir*line.length/2.*extend
        pt2 = midpt - dir*line.length/2.*extend

        lines[i] = Line(pt1[0], pt1[1], pt2[0], pt2[1])
        lines[i].dead = isdead
    
    return lines

def reestimate_model(model, edgelets, threshold=5):
    e_lines = edgelets.lines
    inliers = compute_votes(edgelets, model, threshold) > 0
    e_lines = e_lines[inliers]


    a = e_lines[:, :2]
    b = -e_lines[:, 2]
    est_model = np.linalg.lstsq(a, b)[0]
    return np.concatenate((est_model, [1.]))


def get_contour_lines(image, surface, min_length,use_contours=False):
    lines = []
    polygons = surface.polygons
    if use_contours:
        polygons = surface.contours

    for poly in polygons:
        num_pts = len(poly)

        for i in range(num_pts):
            point_a = poly[i][0]
            point_b = poly[(i+1) % num_pts][0]

            if line_on_image_edge(point_a, point_b, image.shape[1], image.shape[0]):
                continue

            if distance.euclidean(point_a, point_b) > min_length:
                lines.append(Line(point_b[0], point_b[1], point_a[0], point_a[1]))
    
    return lines


class VanishingPoint:
    def __init__(self, lines, model, votes, measure_area=False):

        self.lines = lines
        self.model = model
        self.votes = votes
        self.measure_area = measure_area

        self._score = None
        self._inliers = None
        self._inlier_lines = None

        self._points = None
        #cv2.minAreaRect(InputArray  points)

    def __lt__(self, other):
        return self.score < other.score

    @property
    def score(self):
        if self._score is None:
            self._score = sum(self.votes)

            if self.measure_area and len(self.points) > 1:
                rect = cv2.minAreaRect(np.int32(self.points))

                self._score = self._score * np.sqrt(rect[1][0]*rect[1][1])

        return self._score

    @property
    def inliers(self):
        if self._inliers is None:
            self._inliers = np.array(self.lines)[self.votes > 0]

        return self._inliers
    
    @inliers.setter
    def inliers(self, value:list):
        self._inliers = value

        return self._inliers

    @property
    def points(self):
        if self._points is None:
            self._points = np.array(list(map(lambda x: x.data, self.inliers)), dtype=float).reshape(len(self.inliers) * 2, 2)
        return self._points

    @property
    def direction(self):
        return self.model[:2] / self.model[2]



        
class Edgelets:
    def __init__(self, locations, directions, strengths):
        self.locations = locations
        self.directions = directions
        self.strengths = strengths

        self.normals = np.zeros_like(self.directions)
        self.normals[:, 0] = self.directions[:, 1]
        self.normals[:, 1] = -self.directions[:, 0]
        p = -np.sum(self.locations * self.normals, axis=1)

        self.lines = np.concatenate((self.normals, p[:, np.newaxis]), axis=1)

class Direction(IntEnum):
    Vertical = 0
    Horizontal = 1

class VanishingPointFinder():

    def __init__(self,  edgelets, lines, direction:Direction=None, angle_threshold=np.radians(80)):
        super().__init__()
        self.lines = lines
        self.edgelets = edgelets
        self.direction = direction
        self.angle_threshold = angle_threshold

    def solve(self, max_iterations=2000, threshold_inlier=math.radians(2), max_time=0.33, measure_area=False):

        if self.edgelets is None:
            return []

        num_pts = self.edgelets.strengths.size

        arg_sort = np.argsort(-self.edgelets.strengths)
        first_index_space = arg_sort[:num_pts // min(5, len(self.edgelets.lines))]
        second_index_space = arg_sort[:num_pts // 2]

        self.best_model = None
        self.vanishing_points = []
        
        pi_2 = np.pi/2       

        max_iterations = min(max_iterations, len(self.edgelets.lines) * 40)
        start_time = time.time() 

        for ransac_iter in range(max_iterations):
            if time.time() - start_time > max_time:
                break

            ind1 = np.random.choice(first_index_space)
            ind2 = np.random.choice(second_index_space)

            l1 = self.edgelets.lines[ind1]
            l2 = self.edgelets.lines[ind2]

            current_model = np.cross(l1, l2)

            if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
                # reject degenerate candidates
                continue

            if self.lines[ind1].cluster == self.lines[ind2].cluster:
                # reject lines from the same cluster
                continue


            # if self.direction is not None:
            #     line1 = self.lines[ind1]
            #     line2 = self.lines[ind2]

            #     if self.direction == Direction.Vertical:
            #         if LineFunctions.line_angle_difference(line1.angle, pi_2) > self.angle_threshold or LineFunctions.line_angle_difference(line2.angle, pi_2) > self.angle_threshold:
            #             continue
            #     else:
            #         if LineFunctions.line_angle_difference(line1.angle, 0) > self.angle_threshold or LineFunctions.line_angle_difference(line2.angle, 0) > self.angle_threshold:
            #             continue


            current_model = current_model / current_model[2]

            vp = VanishingPoint(self.lines, current_model, compute_votes(self.edgelets, current_model, threshold_inlier), measure_area=measure_area)
            
            self.vanishing_points.append(vp)

        self.vanishing_points.sort(key=lambda x:x.score, reverse=True)

        return self.vanishing_points

class PipelineVanishingPointFinder(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.VanishingPoints

    @property
    def required_keys(self) -> list:
        return ["room", "downscaled", "isolated", "lines"]

    @property
    def output_keys(self) -> list:
        return ["lines"]

                

    def run(self, data):

        room = data["room"]
        image = data["downscaled"]

        cluster_image = image.copy()
        for i in range(0, len(data["lines"])):
            # print(data["lines"][i].cluster)

            cluster_lines = list(filter(lambda x: x.cluster==i, data["lines"]))
            if len(cluster_lines) > 0:
                # cluster_data = map(lambda x: x.data, cluster_lines)
                # cluster_data = np.array(list(cluster_data))
                # cluster_data = cluster_data.reshape(len(cluster_data) * 2, 2)

                # hull = np.int32(cv2.convexHull(cluster_data))
                # print(hull)
                # cv2.drawContours(cluster_image, [hull], -1, random_color(), -1)

                draw_lines(cluster_image, cluster_lines, color=random_color(), thickness=2)

        log_image(data, "cluster_image", cluster_image)

        diagonal = math.hypot(image.shape[0], image.shape[1])

        #find single vertical vanishing point
        pi_2 = np.pi/2

        vp_lines = []
        room.horizontal_vps = []

        vps_horizontal = []
        vertical_vp = None
        all_lines = data["lines"]
        edgelets = compute_edgelets(all_lines)

        current_indices = [True]*len(all_lines)
        current_line_number = len(all_lines)
        current_lines = all_lines.copy()
        lines_plus = data["lines"].copy()
        all_indices = [True]*len(all_lines)

        # lines_plus.extend(data["hed_lines"])
        # lines_plus.extend(data["normals_lines"])



   
       
        img = image.copy()
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for c in range(1000):
            colors.append(random_color())

        cluster_index = 0

        while cluster_index < 3 if cluster_index < 3 else current_line_number > len(all_lines)/10: 
      
            print("lines in play: ", current_line_number)


            if vertical_vp is None:
                vertical_threshold = np.radians(20)
                vertical_lines, _= partition(lambda x: LineFunctions.line_angle_difference(x.angle, pi_2) < vertical_threshold, data["lines"])

                vertical_indices = list([all_lines[i] in vertical_lines for i in range(len(all_lines))])
                current_indices = vertical_indices
   
                ransac_threshold = np.radians(2.0)
            else:
                ransac_threshold = np.radians(2.5)
                if len(vps_horizontal) == 0:
                    horizontal_threshold = np.radians(.5)
                    horizontal_indices = [False]
                    while np.count_nonzero(horizontal_indices) < 5:
                        horizontal_lines, _= partition(lambda x: LineFunctions.line_angle_difference(x.angle, 0) < horizontal_threshold, data["lines"])

                        horizontal_indices = list([all_lines[i] in horizontal_lines for i in range(len(all_lines))])
                        horizontal_threshold += np.radians(.5)

                    current_indices = horizontal_indices
                    print("no horizontal vps", np.count_nonzero(horizontal_indices))     


            current_lines = list([all_lines[i] for i in range(len(all_lines)) if current_indices[i]])

            current_edgelets = Edgelets(edgelets.locations[current_indices], edgelets.directions[current_indices], edgelets.strengths[current_indices])
            
            vpf = VanishingPointFinder(current_edgelets, current_lines)
            current_vps = vpf.solve(threshold_inlier=ransac_threshold, max_iterations=500,max_time=1.,measure_area=False)

            if len(current_vps) == 0:
                break

            vp = current_vps[0]


            if vertical_vp is None:
                vertical_vp = vp
                
                inliers = list(get_inliers(all_lines, vp.model, 4*ransac_threshold))
                
                for i in range(len(all_lines)):
                    line = all_lines[i]
                    if line in inliers:
                        all_indices[i] = False
                
                # vp.inliers = list(filter(lambda x: not x.dead, vp.inliers))
                # vp.inliers = list(get_vanishing_point_lines(vp.inliers, vp.model,extend=1.0))
                # vp.inliers.sort(key=lambda x:x.angle%np.pi)
                # vp.inliers = merge_lines(vp.inliers, diagonal/400,search_length=2.0,angle_threshold=np.radians(0.5))
                
            else:
                vps_horizontal.append(vp)
          
                # vp.inliers = list(get_inliers(lines_plus, vp.model, ransac_threshold))

                for i in range(len(all_lines)):
                    line = all_lines[i]
                    if line in vp.inliers and line in all_lines:
                        all_indices[i] = False    

                if len(vps_horizontal) > 0:
                    current_indices = all_indices   
                    current_line_number = np.count_nonzero(current_indices)
         
     
                
                # vp.inliers = list(vp.inliers)
                # get_vanishing_point_lines(vp.inliers, vp.model)

                # vp.inliers = list(filter(lambda x: not x.dead, vp.inliers))
                       
                # vp.inliers = merge_lines(vp.inliers, search_length=1.05, search_width=min(diagonal/300, 8), angle_threshold=math.radians(2.0))
                # vp.inliers.sort(key=lambda x:x.angle)

                # get_vanishing_point_lines(vp.inliers, vp.model)
                # vp.inliers = merge_lines(vp.inliers, diagonal/500,search_length=1.0,angle_threshold=np.radians(0.5))

                # vp_pt = reestimate_model(current_vps[0].model, current_edgelets, np.radians(0.5))
                # vp.model = vp_pt

            vp_lines.extend(vp.inliers)


            # draw_lines(img, vp.inliers, color=color, thickness=2,lineType=cv2.LINE_AA)
            color = colors[cluster_index]
            for f in range(len(vp.inliers)):
                l = vp.inliers[f]
                # color =  colors[l.cluster]
                # m = float(f/len(vp.inliers))
                m = 1
                # draw_lines(img, [l], color=(color[0]*m, color[1]*m, color[2]*m), thickness=1,lineType=cv2.LINE_AA)
                vp_pt = vp.model
                vp_pt = vp_pt[:2]/vp_pt[2]
                polyline = np.int32([[l.point_a[0], l.point_a[1]],[l.point_b[0], l.point_b[1]], [vp_pt[0], vp_pt[1]]])
                # cv2.polylines(img, [polyline], False, color=(color[0]*m, color[1]*m, color[2]*m), thickness=1,lineType=cv2.LINE_AA)
                cv2.line(img, (int(l.point_a[0]), int(l.point_a[1])), (int(l.point_b[0]), int(l.point_b[1])), color=(color[0]*m, color[1]*m, color[2]*m), thickness=2,lineType=cv2.LINE_AA)
            # print(cluster_index, color)
            cluster_index+=1



        room.vertical_vp = vertical_vp
        
        room.horizontal_vps = vps_horizontal
        #data["lines"] = vp_lines
        data["vp_lines"] = vp_lines

        log_image(data, "vanishing_pts", img)


        

        # vertical_lines_extended = []
        # vertical_vp_pt = vertical_vp.model[:2]/vertical_vp.model[2]
        
        # surface_lines = data["lines"]
        # vertical_inliers = get_inliers(surface_lines, vertical_vp.model, np.radians(3.0))

        # vertical_inliers = [lines for lines in vertical_inliers if lines in surface_lines]
        # vertical_inliers = merge_lines(vertical_inliers, diagonal/300,angle_threshold=np.radians(0.5))

        # vertical_inliers_directions = [line.midpoint-vertical_vp_pt for line in vertical_inliers]
        # vertical_inliers_directions = vertical_inliers_directions/np.linalg.norm(vertical_inliers_directions,axis=1)[:,np.newaxis]
        
        # vertical_inliers_angles = np.arctan2(np.sign(vertical_inliers_directions[:,1]),vertical_inliers_directions[:,0])
        # vertical_inliers_angles = vertical_inliers_angles
        # angle_ordering = np.argsort(vertical_inliers_angles)
        # vertical_inliers = [vertical_inliers[i] for i in angle_ordering]
        # vertical_inliers_angles = vertical_inliers_angles[angle_ordering]
        # vertical_inliers_angles = np.insert(vertical_inliers_angles,0,vertical_inliers_angles[0] - np.pi/4)
        # vertical_inliers_angles = np.append(vertical_inliers_angles,vertical_inliers_angles[-1] + np.pi/4)

        # for i in range(len(vertical_inliers)):
        #     line = vertical_inliers[i]
        #     color = (255,127/len(vertical_inliers)*i)
        #     draw_lines(img, [line], color=color, thickness=2,lineType=cv2.LINE_AA)



        #     if surface.surfaceType == SurfaceType.Wall or surface.bestLabel in box_like:                
        #         vertical, horizontal = partition(lambda x: LineFunctions.line_angle_difference(x.angle, pi_2) < vertical_threshold, lines)
        #         # horizontal = merge_lines(horizontal, search_width=diagonal/200, search_length=1.1)

        #         vpf = VanishingPointFinder(horizontal)
        #         surface.horizontal_vp = vpf.solve(measure_area=True)
        #         # surface.horizontal_vp = vpf.solve()


        #         vertical_lines.extend(vertical)
        #     else:
        #         vpf = VanishingPointFinder(lines)
        #         surface.vp = vpf.solve(threshold_inlier=math.radians(5), max_iterations=500, max_time=0.2)

        # #find vertical vanishing point for entire room
        # if len(vertical_lines) > 1:
        #     vertical_lines = merge_lines(vertical_lines, search_width=diagonal/200, angle_threshold=math.radians(5))
        #     vpf = VanishingPointFinder(vertical_lines)
        #     room.vertical_vp = vpf.solve(threshold_inlier=np.radians(3), max_time=0.5)
        #     if len(room.vertical_vp) == 0:
        #         room.vertical_vp = vpf.solve(threshold_inlier=np.radians(5))

        # if room.vertical_vp is not None:
        #     for surface in surfaces:
        #         if surface.vertical_vp is None:
        #             surface.vertical_vp = room.vertical_vp

        # if im_logging_enabled(data):
        #     log_image(data, "vanishing_points", self.get_debug_image(data, surfaces, all_lines))

    def get_debug_image(self, data, surfaces, all_lines):
           
        room = data["room"]
        image = data["downscaled"].copy()
    
        #draw all lines
        draw_lines(image, all_lines, color=(80,80,80))

        if room.vertical_vp is not None and len(room.vertical_vp) > 0:
            draw_lines(image, room.vertical_vp[0].inliers, color=(0,255,0), thickness=2)

        for surface in surfaces:
            if surface.horizontal_vp is not None and len(surface.horizontal_vp) > 0:
                draw_lines(image, surface.horizontal_vp[0].inliers, color=random_color(), thickness=2)

            if surface.vp and len(surface.vp):
                draw_lines(image, surface.vp[0].inliers, color=random_color(), thickness=2)
            
        return image

