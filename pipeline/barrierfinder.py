import numpy as np
from scipy import ndimage
import cv2
import random
import time
from scipy.spatial import distance
import math

from .ade20k import ADE20K
from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .utils import resize_array, random_color, overlay_mask, normalize, convert_color, put_text, adjust_mask
from .planegeometry import Dimension
from .extractsurfaces import box_like, legged_objects
from .vanishingpointfinder import angle_with_vp
from .logging import log_image, log_segmentation_image, im_logging_enabled
from cambrian.LineFunctions import LineFunctions
from .Line import line_angle_difference, Line, line_on_image_edge
from .room import Room, Surface

class SurfaceBarriers():
    def __init__(self, data, surface):
        self.data = data
        self.surface = surface
        self.image = self.data["downscaled"]
        self.diagonal = math.hypot(self.image.shape[0], self.image.shape[1])
        self.barrier_candidates = self.get_barrier_candidates()

    def get_barrier_candidates(self):

        def inside_mask(mask, point):
            if point[0] < mask.shape[1] and point[1] < mask.shape[0]:
                return mask[int(point[1]), int(point[0])] > 0
            return False
        
        padding = 10
        surface_mask = cv2.copyMakeBorder(self.surface.mask, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=0) 
        trans = cv2.distanceTransform(1-surface_mask, cv2.DIST_L2, 5)
        _, outer_mask = cv2.threshold(trans, 0.05 * trans.max(), 1, 0)

        trans = cv2.distanceTransform(surface_mask, cv2.DIST_L2, 5)
        _, inner_mask = cv2.threshold(trans, 0.5 * trans.max(), 1, 0)

        self.mask_edges = 1 - inner_mask - outer_mask
        self.mask_edges[self.mask_edges < 0] = 0
        self.mask_edges = self.mask_edges[padding:-padding,padding:-padding]

        barrier_lines = []

        for line in self.data["lines"]:

            if line_on_image_edge(line.point_a, line.point_b, self.image):
                continue

            point_a = (line.point_a[0] + line.midpoint[0]) / 2, (line.point_a[1] + line.midpoint[1]) / 2
            point_b = (line.point_b[0] + line.midpoint[0]) / 2, (line.point_b[1] + line.midpoint[1]) / 2

            if inside_mask(self.mask_edges, line.midpoint) and (inside_mask(self.mask_edges, point_a) or inside_mask(self.mask_edges, point_b)):
                barrier_lines.append(line)                    

        for poly in self.surface.polygons:
            num_pts = len(poly)
            for i in range(num_pts):
                point_a = poly[i][0]
                point_b = poly[(i+1) % num_pts][0]

                if line_on_image_edge(point_a, point_b, self.image):
                    continue

                line = Line(np.array([point_a[0], point_a[1], point_b[0], point_b[1]]))
                barrier_lines.append(line)

        candidates = Line.merge(barrier_lines, search_width=self.diagonal/200, search_length=1.2)

        #filter and correct to vp
        filtered_candidates = []
        for line in candidates:
            
            #check for validity with center:
            #midpoint_angle = LineFunctions.line_angle(self.surface.center[0], self.surface.center[1], line.midpoint[0], line.midpoint[1])

            #check for validity with vanishing point:
            vp_match = next(filter(lambda vp: vp.is_inlier(line, np.radians(5)), self.surface.vanishing_points), None)

            if vp_match is None:
                continue

            filtered_candidates.append(line)

        filtered_candidates = Line.merge(filtered_candidates, search_width=self.diagonal/200)

        return filtered_candidates

    def debug(self, img, color):
        Line.draw_all(img, self.barrier_candidates, color=color, thickness=2) 
        

class BarrierSolver():
    def __init__(self, data, surface):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface = surface


    def solve(self, max_iterations=2000, threshold_inlier=math.radians(2), max_time=0.33, measure_area=False):     

        max_iterations = min(max_iterations, len(self.surface.barriers) * 40)
        start_time = time.time() 

        num_samples = random.randint(2, len(self.barriers))
        for ransac_iter in range(max_iterations):
            if time.time() - start_time > max_time:
                break
        
class PipelineBarrierFinder(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.Barriers

    @property
    def required_keys(self) -> list:
        return ["room", "downscaled", "isolated", "lines"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):

        self.data = data
        self.image = self.data["downscaled"]
        self.room = self.data["room"]
        self.diagonal = math.hypot(self.image.shape[0], self.image.shape[1])

        self.surfaces = []
        self.surfaces.extend(self.room.get_surfaces(surfaceTypes=[SurfaceType.Wall, SurfaceType.WallLike]))
        self.surfaces.extend(self.room.get_surfaces(labels=box_like))

        self.lines = []

        self.barriers = {}

        for surface in self.surfaces:
            self.barriers[surface.uniqueId] = SurfaceBarriers(self.data, surface)

        if im_logging_enabled(self.data):
            log_image(self.data, "barriers.png", self.get_debug_image())


    def get_debug_image(self):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.room.surfaces))

        #overlay probs

        for i in range(len(self.room.surfaces)):
            surface = self.room.surfaces[i]
            
            #mask = (self.barriers[surface.uniqueId][1] if surface.uniqueId in self.barriers else surface.mask) > 0
            mask = surface.mask > 0
            
            max_value = 0.9
            if max_value > 0:
                img_hsv[:, :, 0][mask] = hues[i]
                img_hsv[:, :, 1][mask] = 255 * np.power(surface.probs[mask], 0.15)
                    
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)

        Line.draw_all(img, self.data["lines"], color=(255,255,255), thickness=1)

        for i in range(len(self.room.surfaces)):
            surface = self.room.surfaces[i]
            color = convert_color((hues[i],127,255), cv2.COLOR_HSV2RGB_FULL)

            if surface.uniqueId in self.barriers:
                self.barriers[surface.uniqueId].debug(img, color)

        return img
