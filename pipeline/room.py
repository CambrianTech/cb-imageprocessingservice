from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np
import cv2
import uuid
import random

import warnings #skimage warnings excessive:
warnings.filterwarnings("ignore")

from skimage.segmentation import watershed
from skimage.color import rgb2gray
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial import distance

from .geometry import Geometry
from .core import SurfaceType
from .surface import Surface
from .utils import convert_color, put_text, overlay_mask, random_color
from .logging import im_logging_enabled, log_image, log_segmentation_image, log_markers
from .Line import line_angle_difference, Line, on_image_edge
from .ade20k import ADE20K

from termcolor import colored

class Barrier():
    def __init__(self, data, midpoint, angle=0):
        self.data = data
        self.midpoint = midpoint
        self.angle = angle

class Vertex():
    def __init__(self, center, line_a, line_b):
        self.center = center
        self.line_a = line_a
        self.line_b = line_b
        self.radius = int(min(min(self.line_a.length, self.line_b.length), 30))

    def get_samples(self, image, outside=False):

        x_min, x_max = self.center[0] - self.radius, self.center[0] + self.radius
        y_min, y_max = self.center[1] - self.radius, self.center[1] + self.radius

        x_offset = 0
        if x_min < 0: 
            x_offset = x_min
            x_min = 0
        elif x_max >= image.shape[1]: 
            x_offset = image.shape[1] - x_max + 1
            x_max = image.shape[1] - 1

        y_offset = 0
        if y_min < 0: 
            y_offset = y_min
            y_min = 0
        elif y_max >= image.shape[0]: 
            y_offset = image.shape[0] - y_max + 1
            y_max = image.shape[0] - 1

        mask = np.zeros((y_max - y_min, x_max - x_min), dtype=np.uint8)

        if outside:
            start_angle = self.line_b.angle
            stop_angle = self.line_a.angle + 2 * np.pi
        else:
            start_angle = self.line_a.angle
            stop_angle = self.line_b.angle

        cv2.ellipse(mask, (self.radius + x_offset, self.radius + y_offset), (self.radius, self.radius), 0, np.degrees(start_angle), np.degrees(stop_angle), [255, 255, 255], thickness=cv2.FILLED)

        image_arc = image[y_min:y_max, x_min:x_max][mask > 0]

        segments, counts = np.unique(image_arc, return_counts=True)
        sorted_labels = sorted(zip(segments.tolist(), counts.tolist()), key=lambda x:-x[1])
        
        return sorted_labels


#python info on object oriented methods and properties
#https://stackoverflow.com/questions/2736255/abstract-attributes-in-python
class Room(Geometry):

    def analyze(self):

        self.barrier_contours = []
        self.barrier_candidates = []

        log_segmentation_image(self.data, "semantic_labels", self.semantic_labels, self.image)
        
        self.analyze_surfaces()
        log_image(self.data, "room_initial", self.get_debug_image())

        self.refine_surfaces(min_confidence=0.1) #preserve plane context information i.e. probs < min_confidence are ignored
        log_image(self.data, "room_refined", self.get_debug_image())

        self.add_missing_surfaces()
        log_image(self.data, "room_modified", self.get_debug_image())

        self.refine_surfaces()
        #log_image(self.data, "room_refined_again", self.get_debug_image())
        #self.ransac_fit()

        self.merge_like_surfaces()

        self.find_barriers()

        log_image(self.data, "room", self.get_debug_image())

    def analyze_surfaces(self):
        #perform initial analysis
        for surface in self.surfaces:
            surface.analyze()

    def get_clusters(self, x, kmin=2, kmax=5):
        sil = []
        for k in range(kmin, kmax+1):
          kmeans = KMeans(n_clusters = k).fit(x)
          labels = kmeans.labels_
          sil.append(silhouette_score(x, labels, metric = 'euclidean'))

    def find_best_candidate(self, surfaceType, mask):
        candidates = self.get_surfaces(surfaceType=surfaceType)
        if len(candidates) == 0:
            return None

        if len(candidates) > 1:
            #todo: sort if more than one, for walls, above and below the wall is the best match
            print("Find match for missing %s amongst %d candidates" % (surfaceType.name, len(candidates)))
            normal = np.mean(self.data["normals"], axis=(0, 1)) #todo: use 3d vector normal angle difference instead.
            candidates.sort(key=lambda x: distance.sqeuclidean(normal, x.normals_color))

        return candidates[0]

    def add_missing_surfaces(self, min_area=1/1200):

        total_area = self.image.shape[0] * self.image.shape[1]
        area_threshold = int(total_area * min_area)

        print("Size threshold: square greater than %d pixels on one side" % np.sqrt(area_threshold))

        if len(self.surfaces) > 0:
            total_mask = np.sum(np.dstack([s.mask for s in self.surfaces]), axis=-1)
        else:
            total_mask = None

        room_missing = self.image.copy() if im_logging_enabled(self.data) else None

        total_elevation = 3 #todo: get total elevation from highest and lowest objects. Floor or ceiling could be missing
        
        for surfaceType in SurfaceType:
            
            color = random_color()

            #find missing
            remaining_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
            remaining_mask[self.isolated_labels == surfaceType] = 1

            if total_mask is not None:
                remaining_mask[total_mask > 0] = 0

            contours, hierarchy = cv2.findContours(remaining_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = []
            if contours is not None:
                for contour in contours:
                    if cv2.contourArea(contour) > area_threshold:
                        valid_contours.append(contour)

            #draw
            if room_missing is not None:
                cv2.drawContours(room_missing, np.array(valid_contours), -1, color, cv2.FILLED)
                log_image(self.data, "room_missing", room_missing)

            #add missing
            for contour in valid_contours:

                contour_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], 0, (1,1,1), cv2.FILLED)

                matches = self.index_mask[contour_mask > 0]
                indexes, counts = np.unique(matches, return_counts=True)
                
                valid_clusters = indexes[counts > area_threshold]

                max_clusters = len(valid_clusters)
                new_surface = None
                reference_surface = None

                for i in range(max_clusters):
                    index = valid_clusters[i]
                    surface = self.surfaces[index]

                    if surface.surfaceType == surfaceType or surface.surfaceType.is_pair(surfaceType):

                        mask = contour_mask.copy()
                        mask[self.index_mask != index] = 0

                        if cv2.countNonZero(mask) >= area_threshold:
                            reference_surface = surface
                            new_surface = reference_surface.clone()
                            new_surface.surfaceType = surfaceType
                            new_surface.set_mask(mask)
                            break

                    elif not surface.bestLabel:
                        surface.destroyed = True
                        break

                new_normal = None
                new_offset = None

                if new_surface is None and surfaceType.is_major:
                    reference_surface = self.find_best_candidate(surfaceType, contour_mask)
                    
                    if reference_surface is not None:
                        new_surface = reference_surface.clone()
                        new_surface.surfaceType = surfaceType
                        new_surface.set_mask(contour_mask)
                    elif surfaceType == SurfaceType.Floor or surfaceType == SurfaceType.Ceiling:
                        complimentary_type = SurfaceType.Floor if surfaceType == SurfaceType.Ceiling else SurfaceType.Ceiling
                        reference_surface = self.find_best_candidate(complimentary_type, contour_mask)

                        if reference_surface is not None:
                            print("Generate %s using %s as opposing surface" % (surfaceType.name, reference_surface.name))
                            
                            new_surface = reference_surface.clone()
                            new_surface.surfaceType = surfaceType
                            new_surface.set_mask(contour_mask)
                            new_surface.normal = -reference_surface.normal
                            new_surface.offset = total_elevation - reference_surface.offset


                    if reference_surface is None:
                        print(colored("No match for %s" % surfaceType.name, 'yellow'))
                        

                if new_surface is not None:
                    print(colored("Creating new %s using %s as reference" % (surfaceType.name, reference_surface.name), 'green'))
                    #log_image(self.data, new_surface.name, new_surface.mask * 255)
                    self.add_surface(new_surface)
            

    def merge_like_surfaces(self, angle_threshold=np.radians(20)):

        for surfaceType in SurfaceType:
            
            color = random_color()
            surfaces = self.get_surfaces(surfaceType)


            for i in range(len(surfaces)):

                if surfaces[i].destroyed: continue

                distance_i = abs(surfaces[i].offset) #todo: calculate this?

                for j in range(i+1, len(surfaces)):

                    if surfaces[j].destroyed: continue

                    dot_product = np.dot(surfaces[i].normal, surfaces[j].normal)
                    angle = np.arccos(dot_product)

                    #do some planar geometry comparisons, maybe color/texture
                    distance_j = abs(surfaces[j].offset)

                    distance_between = abs(distance_i - distance_j)
                    distance_mean = 0.5 * (distance_i + distance_j)
                    distance_error = 0.35 * distance_mean #accuracy degrades by range (maybe use error here, error square?)

                    #todo: check for intersection. In elevator image, wall sitting out front is being incorrectly merged. if it's fairly parallel, don't
                    if surfaceType == SurfaceType.Floor or surfaceType == SurfaceType.Ceiling or (angle < angle_threshold and distance_between < distance_error):
                        if surfaceType != SurfaceType.Other or surfaces[i].bestLabel == surfaces[j].bestLabel:
                            surfaces[i].merge(surfaces[j])
                            surfaces[i]._alteration = "%.2fm %.2fd" % (distance_between, angle_threshold)

        
    def refine_surfaces(self, min_confidence=None, use_lines=True):
        watershed_image = cv2.resize(self.data["hed"], (self.image.shape[1], self.image.shape[0]))
        
        if use_lines:
            lines_mask = np.zeros(watershed_image.shape, dtype=np.uint8)
            Line.draw_all(lines_mask, self.data["lines"], color=(255,255,255), thickness=1, lineType=cv2.LINE_4)

        def expand_into_type(surfaceType:SurfaceType):
            surfaces = self.get_surfaces(surfaceType)

            num_surfaces = len(surfaces)

            if num_surfaces == 0:
                return

            total_mask = np.sum(np.dstack([s.mask for s in surfaces]), axis=-1)
            disputed_areas = np.zeros(total_mask.shape, dtype=np.uint8)
            disputed_areas[total_mask > 1] = 1

            markers = np.zeros(total_mask.shape, dtype=np.int32)
            
            for index in range(num_surfaces):
                surface = surfaces[index]
                dist_transform = cv2.distanceTransform(surface.mask, distanceType=cv2.DIST_L2, maskSize=3, dstType=cv2.CV_8U)
                markers[dist_transform > 0.15 * dist_transform.max()] = index + 1

            markers[disputed_areas > 0] = 0

            watershed_mask = np.zeros(total_mask.shape, dtype=np.int32)
            watershed_mask[self.isolated_labels == surfaceType.index] = 1
            if use_lines:
                watershed_mask[lines_mask > 0] = 0

            log_markers(self.data, "room_%s_markers" % surfaceType.name, markers, mask=watershed_mask)

            #perform watershed:
            markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))
            markers[markers<0] = 0

            log_markers(self.data, "room_%s_watershed" % surfaceType.name, markers, mask=watershed_mask)

            #commit to mask
            for index in range(num_surfaces):
                surface = surfaces[index]
                mask = np.zeros_like(surface.mask)
                mask[markers == (index + 1)] = 1
                if min_confidence is not None: 
                    mask[surface.probs < 0.01] = 0

                surface.set_mask(mask)


        #expand all surfaces as far as they can go within their segmentation (watershed)
        #and resolve disputes between planes as they intersect by probability (confidence):
        for surfaceType in SurfaceType: 
            expand_into_type(surfaceType)

    def find_barriers(self):
        
        #ceilings do not have as many things on them, so iterate across contours, looking for points downward
        #images may lack floors, ceilings, or both

        distance_check = self.image.shape[0] / 30
        min_length = self.image.shape[0] / 50

        self.barrier_contours = []
        self.barrier_candidates = []

        def find_candidates(surface, poly, min_angle_threshold, max_angle_threshold):
            num_pts = len(poly)
            candidates = []

            for i in range(num_pts):
                point_a = poly[i][0]
                point_b = poly[(i+1) % num_pts][0]
                point_c = poly[(i+2) % num_pts][0]

                if on_image_edge(point_b, self.image):
                    continue

                line_a = Line(np.array([(point_b[0], point_b[1], point_a[0], point_a[1])], dtype=np.int).reshape(4))
                line_b = Line(np.array([(point_b[0], point_b[1], point_c[0], point_c[1])], dtype=np.int).reshape(4))

                if line_a.length < min_length or line_b.length < min_length:
                    continue

                angle = line_angle_difference(line_a.angle, line_b.angle)
                if angle >= min_angle_threshold and angle <= max_angle_threshold:
                    #check for type differential of the labels inside an arc (see debug arc):
                    vertex = Vertex(point_b, line_a, line_b)
                    
                    inner_labels = vertex.get_samples(self.isolated_labels)
                    
                    if len(inner_labels) > 0:
                        best_label, best_count = inner_labels[0]
                        if len(inner_labels) < 3 and best_label == surface.surfaceType.index:
                            outer_labels = vertex.get_samples(self.isolated_labels, outside=True)
                            best_outer, best_outer_count = outer_labels[0]

                            #if nowhere near a wall, forget it
                            if best_label != SurfaceType.Wall.index and best_label != SurfaceType.WallLike.index \
                                and best_outer != SurfaceType.Wall.index and best_outer != SurfaceType.WallLike.index:
                                continue

                            if len(inner_labels) > 1:
                                #remove clutter
                                second_best_label, second_best_count = inner_labels[1]
                                if best_count / second_best_count > 3:
                                    candidates.append(vertex)   
                            else:
                                candidates.append(vertex)

            return candidates

        
        def build_barriers(surfaceType, min_angle_threshold=np.radians(30), max_angle_threshold=np.radians(170)):
            for surface in self.get_surfaces(surfaceType):
                for poly in surface.polygons:
                    self.barrier_contours.append(poly)
                    self.barrier_candidates.extend(find_candidates(surface, poly, min_angle_threshold, max_angle_threshold))
                
        
        build_barriers(SurfaceType.Ceiling)
        build_barriers(SurfaceType.Floor)
        build_barriers(SurfaceType.Wall)
        build_barriers(SurfaceType.WallLike)


            #contours = ceilings.concatenate

        
    def get_debug_image(self):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.surfaces))

        #overlay probs
        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            mask = surface.mask > 0

            max_value = 0.9
            if max_value > 0:
                img_hsv[:, :, 0][mask] = hues[i]
                img_hsv[:, :, 1][mask] = 255 * np.power(surface.probs[mask], 0.15)
                    
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)

        #let surface do its debug
        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            hue = hues[i]
            color = convert_color((hue, 255, 255), cv2.COLOR_HSV2RGB_FULL)
            surface.debug(img, color)

        #cv2.drawContours(img, self.barrier_contours, -1, color=(255,255,0), thickness=1) 

        Line.draw_all(img, self.data["lines"], color=(80,80,80), thickness=2)

        for vertex in self.barrier_candidates:

            cv2.ellipse(img, vertex.center, (vertex.radius, vertex.radius), 0, np.degrees(vertex.line_a.angle), np.degrees(vertex.line_b.angle), [0, 255, 0], thickness=1) 
            #cv2.ellipse(img, vertex.center, (vertex.radius, vertex.radius), 0, np.degrees(vertex.line_b.angle), np.degrees(vertex.line_a.angle) + 360, [255, 0, 0], thickness=2) 

            cv2.line(img, vertex.line_a.point_a, vertex.line_a.point_b, [0, 0, 255], thickness=2)
            cv2.line(img, vertex.line_b.point_a, vertex.line_b.point_b, [255, 255, 0], thickness=2)

        for vertex in self.barrier_candidates:
            cv2.drawMarker(img, vertex.center, color=(255,0,0), thickness=2)


        return img


