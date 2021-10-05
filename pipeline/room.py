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

from .geometry import Geometry
from .core import SurfaceType
from .surface import Surface
from .utils import convert_color, put_text, overlay_mask, random_color
from .logging import im_logging_enabled, log_image, log_segmentation_image, log_markers
from .Line import Line
from .ade20k import ADE20K

from termcolor import colored

#python info on object oriented methods and properties
#https://stackoverflow.com/questions/2736255/abstract-attributes-in-python
class Room(Geometry):

    @property
    def ceilings(self):
        return self.get_surfaces(surfaceType=SurfaceType.Ceiling)

    @property
    def walls(self):
        return self.get_surfaces(surfaceType=SurfaceType.Wall)

    @property
    def floors(self):
        return self.get_surfaces(surfaceType=SurfaceType.Floor)

    def analyze(self):

        log_segmentation_image(self.data, "semantic_labels", self.semantic_labels, self.image)
        
        self.analyze_surfaces()
        log_image(self.data, "room_initial", self.get_debug_image())

        self.refine_surfaces(min_confidence=0.1) #preserve plane context information i.e. probs < 0.1 are ignored
        log_image(self.data, "room_refined", self.get_debug_image())

        self.add_missing_surfaces()
        log_image(self.data, "room_modified", self.get_debug_image())

        self.refine_surfaces()
        #log_image(self.data, "room_refined_again", self.get_debug_image())
        #self.ransac_fit()

        self.merge_like_surfaces()

        log_image(self.data, "room", self.get_debug_image())

    @property
    def semantic_type() -> ADE20K:
        #todo: get value
        return ADE20K.shelf


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


    def add_missing_surfaces(self):

        total_area = self.image.shape[0] * self.image.shape[1]
        area_threshold = total_area / 300

        if len(self.surfaces) > 0:
            total_mask = np.sum(np.dstack([s.mask for s in self.surfaces]), axis=-1)
        else:
            total_mask = None

        if im_logging_enabled(self.data):
            debug = self.image.copy()
        
        for surfaceType in SurfaceType:
            
            color = random_color()

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

            if im_logging_enabled(self.data) and len(valid_contours):
                cv2.drawContours(debug, np.array(valid_contours), -1, color, cv2.FILLED)

            for contour in valid_contours:
                contour_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], 0, (1,1,1), cv2.FILLED)

                matches = self.index_mask[contour_mask > 0]
                indexes, counts = np.unique(matches, return_counts=True)
                
                valid_clusters = indexes[counts > area_threshold]

                max_clusters = len(valid_clusters)
                new_surface = None

                for i in range(max_clusters):
                    index = valid_clusters[i]
                    surface = self.surfaces[index]

                    if surface.surfaceType == surfaceType or surface.surfaceType.is_pair(surfaceType):

                        mask = contour_mask.copy()
                        mask[self.index_mask != index] = 0

                        if cv2.countNonZero(mask) >= area_threshold:
                            new_surface = surface.clone()
                            new_surface.surfaceType = surfaceType
                            new_surface.set_mask(mask)
                            break
                    elif not surface.bestLabel:
                        surface.destroyed = True
                        break

                if new_surface is not None:
                    print(colored("Creating new %s using %s as reference" % (surfaceType.name, surface.name), 'green'))
                    log_image(self.data, surface.name, new_surface.mask)
                    self.add_surface(new_surface)

        if im_logging_enabled(self.data):
            log_image(self.data, "room_missing", debug)

    def merge_like_surfaces(self, angle_threshold=np.radians(30)):

        if im_logging_enabled(self.data):
            debug = self.image.copy()


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

                    if surfaceType == SurfaceType.Floor or (angle < angle_threshold and distance_between < distance_error):
                        if surfaceType != SurfaceType.Other or surfaces[i].bestLabel == surfaces[j].bestLabel:
                            surfaces[i].merge(surfaces[j])
                            surfaces[i]._alteration = "%.2fm %.2fd" % (distance_between, angle_threshold)


        if im_logging_enabled(self.data):
            log_image(self.data, "room_merged", debug)

        
    def refine_surfaces(self, min_confidence=None):
        watershed_image = cv2.resize(self.data["hed"], (self.image.shape[1], self.image.shape[0]))
        lines_mask = np.zeros(watershed_image.shape, dtype=np.uint8)
        sx = self.image.shape[1] / self.data["image"].shape[1]
        sy = self.image.shape[0] / self.data["image"].shape[0]
        Line.draw_all(lines_mask, self.data["lines"], color=(255,255,255), thickness=1, sx=sx, sy=sy, lineType=cv2.LINE_4)

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

    def ransac_fit(self):
        pass

        
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
                img_hsv[:, :, 1][mask] = 255 * np.power(surface.probs[mask], 0.25)
                    
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)

        #let surface do its debug
        for i in range(len(self.surfaces)):
            surface = self.surfaces[i]
            hue = hues[i]
            color = convert_color((hue, 255, 255), cv2.COLOR_HSV2RGB_FULL)

            surface.debug(img, color)
                
        sx = self.image.shape[1] / self.data["image"].shape[1]
        sy = self.image.shape[0] / self.data["image"].shape[0]
        Line.draw_all(img, self.data["lines"], color=(0,0,255), thickness=1, sx=sx, sy=sy)

        return img


