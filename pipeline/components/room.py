from abc import ABCMeta, abstractmethod
import numpy as np
import cv2
import uuid
import random
import math

import warnings #skimage warnings excessive:
warnings.filterwarnings("ignore")

from skimage.segmentation import watershed
from skimage.color import rgb2gray
from scipy.stats import mode
from scipy.spatial import distance

from .geometry import Geometry
from pipeline.data.surface_type import SurfaceType
from .surface import Surface
from pipeline.misc.utils import convert_color, put_text, overlay_mask, random_color, sample_at_point, scale_contour
from pipeline.data.logging import im_logging_enabled, log_image, log_segmentation_image, log_markers, Timer
from .line import line_angle_difference, Line, draw_lines
from pipeline.data.ade20k import ADE20K

from termcolor import colored

#python info on object oriented methods and properties
#https://stackoverflow.com/questions/2736255/abstract-attributes-in-python

def narrowness(contour, epsilon_factor=0.06):
    peri = cv2.arcLength(contour, True)
    epsilon = epsilon_factor * peri
    hull = cv2.convexHull(contour)
    points = cv2.approxPolyDP(hull, epsilon, True)
    
    if len(points) < 3:
        return np.inf

    area = cv2.contourArea(points)
    square_side_length = math.sqrt(area)

    num_points = len(points)
    
    #use np.diff or something better
    lengths = []
    for i in range(num_points):
        point_a = points[i][0]
        point_b = points[(i+1) % num_points][0]
        length = distance.euclidean(point_a, point_b)
        if length > square_side_length:
            lengths.append(distance.euclidean(point_a, point_b))

    lengths = np.array(lengths)
    #mean_length = np.mean(lengths)
    total_length = np.sum(lengths)
    #mode_length = mode(lengths).mode[0]

    return total_length / square_side_length

class Room(Geometry):

    def analyze(self):

        timer = Timer("room")

        #prepare
        self.lines_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
        draw_lines(self.lines_mask, self.data["lines"], color=(255,255,255), thickness=1, lineType=cv2.LINE_4)

        log_segmentation_image(self.data, "semantic_labels", self.semantic_labels, self.image, labelset=ADE20K)

        timer.time_event("setup")

        self.analyze_surfaces()
        log_image(self.data, "room_initial", self.get_debug_image())

        timer.time_event("analyze_surfaces")

        self.refine_surfaces(min_confidence=0.1) #preserve plane context information i.e. probs < min_confidence are ignored
        log_image(self.data, "room_refined", self.get_debug_image())

        timer.time_event("refine_surfaces")

        invalid_mask = self.remove_invalid_surfaces()
        
        if im_logging_enabled(self.data) and cv2.countNonZero(invalid_mask) > 50:
            debug = self.image.copy()
            debug[invalid_mask > 0] = [0,255,0]
            log_image(self.data, "room_removed", debug)

        timer.time_event("remove_invalid_surfaces")

        self.add_missing_surfaces(invalid_mask)
        log_image(self.data, "room_missing_added", self.get_debug_image())

        timer.time_event("add_missing_surfaces")

        self.refine_surfaces(debug_suffix="_final")
        log_image(self.data, "room_refined_again", self.get_debug_image())

        timer.time_event("room.refine_surfaces")

        self.merge_like_surfaces()

        self.finalize_masks(invalid_mask)

        log_image(self.data, "room", self.get_debug_image())

        timer.time_event("merge_like_surfaces")

        self.assign_parents()

        timer.time_event("assign_parents")

        timer.log_all_events()

    def analyze_surfaces(self):
        #perform initial analysis
        for surface in self.surfaces:
            surface.analyze()
        

    def find_best_surface(self, surfaceType, mask, mask_center):

        candidates = self.get_surfaces([surfaceType])
        
        if len(candidates) == 0:
            return None

        if len(candidates) > 1:

            mask_sample = sample_at_point(mask, point=mask_center)

            if cv2.countNonZero(mask_sample) > 0:
                sample = sample_at_point(self.normals, point=mask_center)
                normal = cv2.mean(sample, mask_sample)[:3]
                candidates.sort(key=lambda x: distance.sqeuclidean(mask_center, x.center) * distance.sqeuclidean(normal, x.normals_color))
            else:
                candidates.sort(key=lambda x: distance.sqeuclidean(mask_center, x.center))

        return candidates[0]

    def add_missing_surfaces(self, invalid_mask=None, min_area=1/1200):

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
            if invalid_mask is not None:
                remaining_mask[invalid_mask > 0] = 0

            if total_mask is not None:
                remaining_mask[total_mask > 0] = 0

            contours, hierarchy = cv2.findContours(remaining_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            valid_contours = []
            if contours is not None:
                for contour in contours:
                    area = cv2.contourArea(contour)
                    if area > area_threshold:
                        valid_contours.append(contour)

            #draw
            if room_missing is not None and len(valid_contours) > 0:
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

                #add missing one by one
                for i in range(max_clusters):
                    index = valid_clusters[i]
                    if index >= len(self.surfaces):
                        continue

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

                    moments = cv2.moments(contour_mask)

                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])

                    reference_surface = self.find_best_surface(surfaceType, contour_mask, (cX, cY))
                    
                    if reference_surface is not None:
                        new_surface = reference_surface.clone()
                        new_surface.surfaceType = surfaceType
                        new_surface.set_mask(contour_mask)
                        #print("Reference_surface", reference_surface.name, indexes, counts)
                    elif surfaceType == SurfaceType.Floor or surfaceType == SurfaceType.Ceiling:
                        complementary_type = SurfaceType.Floor if surfaceType == SurfaceType.Ceiling else SurfaceType.Ceiling
                        reference_surface = self.find_best_surface(complementary_type, contour_mask, (cX, cY))

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
                    print(colored("Creating new %s (%s) using %s as reference" % (surfaceType.name, new_surface.name, reference_surface.name), 'green'))
                    self.add_surface(new_surface)
                    

        self.invalidate()

    def merge_like_surfaces(self, angle_threshold=np.radians(30), angle_threshold_force=np.radians(20)):

        for surfaceType in SurfaceType:
                
            color = random_color()
            surfaces = self.get_surfaces([surfaceType])


            for i in range(len(surfaces)):

                if surfaces[i].destroyed: continue

                distance_i = abs(surfaces[i].offset) #todo: calculate this?

                for j in range(i+1, len(surfaces)):

                    if surfaces[j].destroyed: continue

                    if surfaces[i].bestLabel != surfaces[j].bestLabel: continue

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
                            offset_diff = abs(surfaces[i].offset - surfaces[j].offset) / max(abs(surfaces[i].offset), abs(surfaces[j].offset))
                            
                            #print("angle", angle, "offset", offset_diff)

                            if offset_diff < 0.05 or angle < angle_threshold_force: 
                                surfaces[i].merge(surfaces[j])
                                surfaces[i]._alteration = "%.2fm %.2fd" % (distance_between, angle_threshold)

        
    def refine_surfaces(self, min_confidence=None, use_lines=True, debug_suffix=""):
        watershed_image = cv2.resize(self.data["hed"], (self.image.shape[1], self.image.shape[0]))

        final_masks = {}
        
        def expand_into_type(surfaceType:SurfaceType):
            surfaces = self.get_surfaces([surfaceType])

            num_surfaces = len(surfaces)

            if num_surfaces == 0:
                return

            total_mask = np.sum(np.dstack([s.mask for s in surfaces]), axis=-1)
            disputed_areas = np.zeros(total_mask.shape, dtype=np.uint8)
            disputed_areas[total_mask > 1] = 1

            markers = np.zeros(total_mask.shape, dtype=np.int32)
            
            for index in range(num_surfaces):
                surface = surfaces[index]
                dist_transform = surface.mask_transform
                markers[dist_transform > 0.15 * dist_transform.max()] = index + 1

            markers[disputed_areas > 0] = 0

            watershed_mask = np.zeros(total_mask.shape, dtype=np.int32)
            watershed_mask[self.isolated_labels == surfaceType.index] = 1
            if use_lines:
                watershed_mask[self.lines_mask > 0] = 0

            log_markers(self.data, "room_%s_markers%s" % (surfaceType.name, debug_suffix), markers, mask=watershed_mask)

            #perform watershed:
            markers = np.int32(watershed(watershed_image, markers, mask=watershed_mask))
            markers[markers<0] = 0

            log_markers(self.data, "room_%s_watershed%s" % (surfaceType.name, debug_suffix), markers, mask=watershed_mask)

            #commit to mask
            for index in range(num_surfaces):
                surface = surfaces[index]
                mask = np.zeros_like(surface.mask)
                mask[markers == (index + 1)] = 1
                mask[self.lines_mask > 0] = 0
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
                mask = cv2.dilate(mask, kernel)

                if min_confidence is not None: 
                    mask[surface.probs < 0.01] = 0

                surface.set_mask(mask)

        #expand all surfaces as far as they can go within their segmentation (watershed)
        #and resolve disputes between planes as they intersect by probability (confidence):

        for surfaceType in SurfaceType: 
            expand_into_type(surfaceType)

        

    def remove_invalid_surfaces(self, min_area_threshold=1/1000, max_area_threshold=1/50, scale=1.2):

        total_area = self.image.shape[0] * self.image.shape[1]
        max_area = total_area * max_area_threshold
        min_area = total_area * min_area_threshold
        
        all_invalid_contours = []
        print("Remove invalid wall parts.")

        for surface in self.get_surfaces([SurfaceType.Wall]):
            invalid_contours = []

            if surface.surfaceType == SurfaceType.Wall:
                neighboring_walls = list(filter(lambda s:s.surfaceType == SurfaceType.Wall, surface.neighbors))
                print("Surface %s has neighbors:" % surface.name, [neighbor.name for neighbor in neighboring_walls])

            for i in range(len(surface.contours)):
                contour = surface.contours[i]

                M = cv2.moments(contour)
                area = M['m00']

                if area > max_area:
                    continue

                name = "Surface %s(%d)" % (surface.name, i)

                if area < min_area:
                    invalid_contours.append(contour)
                else:
                    inner_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(inner_mask, np.array(contour), 0, 1, cv2.FILLED)
                    
                    innerMean, innerStd = cv2.meanStdDev(self.image, mask=inner_mask)

                    outer_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                    contour_expanded = scale_contour(contour, scale, moments=M)
                    cv2.drawContours(outer_mask, np.array(contour_expanded), 0, 1, cv2.FILLED)
                    #outer_mask[inner_mask] = 0
                    outerMean, outerStd = cv2.meanStdDev(self.image, mask=outer_mask)

                    meanDiff = np.max(np.abs(innerMean - outerMean))

                    #really want the standard deviation here, but opencv is returning ZEROS:
                    #threshold = meanDiff + innerStd * meanDiff

                    print("surface %s(%d) %.2f" % (surface.name, i, meanDiff))

                    if meanDiff < 10:
                        #print("remove %s" % name, meanDiff)
                        invalid_contours.append(contour)
                   
            all_invalid_contours.extend(invalid_contours)

            if len(invalid_contours) == len(surface.contours):
                surface.destroy() #totally invalid
                print(colored("Removing surface %s" % surface.name, "red"))
            elif len(invalid_contours) > 0:
                print(colored("Removing %d contours from surface %s" % (len(invalid_contours), surface.name), "yellow"))
                mask = surface.mask.copy()
                cv2.drawContours(mask, np.array(invalid_contours), -1, 0, cv2.FILLED)
                surface.set_mask(mask)

                #todo: look inside contour for validity

        invalid_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        if len(all_invalid_contours) > 0:
            cv2.drawContours(invalid_mask, np.array(all_invalid_contours), -1, 1, cv2.FILLED)
            self.refresh_surfaces()

        return invalid_mask

    def finalize_masks(self, invalid_mask=None):

        total_mask = []
        markers = np.zeros((self.image.shape[0], self.image.shape[1]), dtype=np.int32)
        if invalid_mask is not None:
            markers[invalid_mask > 0] = -1

        num_surfaces = len(self.surfaces)
        for index in range(num_surfaces):
            surface = self.surfaces[index]
            markers[surface.mask > 0] = index + 1

        markers = cv2.watershed(self.image, markers)
        markers[markers<0] = 0

        for index in range(num_surfaces):
            surface = self.surfaces[index]
            mask = np.zeros_like(surface.mask)
            mask[markers == (index + 1)] = 1
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            surface.set_mask(mask)

        log_markers(self.data, "room_final_markers", markers)

    def assign_parents(self):

        child_surfaces = self.get_surfaces(surfaceTypes=[SurfaceType.OnWall, SurfaceType.OnFloor, SurfaceType.OnCeiling, SurfaceType.Other])

        for child_surface in child_surfaces:

            candidates = child_surface.neighbors.copy() if child_surface.surfaceType.complement is None else list(filter(lambda x: x.surfaceType == child_surface.surfaceType.complement, child_surface.neighbors))
            candidates = list(filter(lambda x: x.parent is None, candidates))

            if len(candidates) == 0:
                continue
            elif len(candidates) == 1:
                child_surface.parent = candidates[0]
                continue

            #sort by most interior. May want to look at vanishing points or just lines clustering in angle.
            candidates.sort(key=lambda x:distance.sqeuclidean(child_surface.center, x.center))

            child_surface.parent = candidates[0]
        
    def get_debug_image(self):
        if not im_logging_enabled(self.data): 
            return None

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

        return img


