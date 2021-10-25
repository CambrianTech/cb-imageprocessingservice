import numpy as np
from scipy import ndimage
import cv2
from skimage.morphology import remove_small_objects
import random
import time

from .ade20k import ADE20K
from .core import PipelineStep, PipelineStepIndex, SurfaceType
from .utils import resize_array, random_color, overlay_mask
from .planegeometry import Dimension
from .extractsurfaces import box_like
from .vanishingpointfinder import draw_vp, angle_with_vp
from .logging import log_image, log_segmentation_image, im_logging_enabled
from cambrian.LineFunctions import LineFunctions
from .Line import line_angle_difference, Line, on_image_edge
from .room import Room, Surface

class SurfaceVertices():
    def __init__(self, surface, vertices):
        self.surface = surface
        self.vertices = vertices

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

class BarrierFinder():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]

    def solve(self):
        
        #ceilings do not have as many things on them, so iterate across contours, looking for points downward
        #images may lack floors, ceilings, or both

        distance_check = self.image.shape[0] / 30
        min_length = self.image.shape[0] / 50

        self.barrier_candidates = []

        def find_vertices(surface, poly, min_angle_threshold, max_angle_threshold):
            num_pts = len(poly)
            candidates = []

            for i in range(num_pts):
                point_a = poly[i][0]
                point_b = poly[(i+1) % num_pts][0]
                point_c = poly[(i+2) % num_pts][0]

                # if on_image_edge(point_a, self.image) and on_image_edge(point_c, self.image):
                #     continue

                line_a = Line(np.array([(point_b[0], point_b[1], point_a[0], point_a[1])], dtype=np.int).reshape(4))
                line_b = Line(np.array([(point_b[0], point_b[1], point_c[0], point_c[1])], dtype=np.int).reshape(4))

                if line_a.length < min_length or line_b.length < min_length:
                    continue

                angle = line_angle_difference(line_a.angle, line_b.angle)

                if angle < min_angle_threshold or angle > max_angle_threshold:
                    continue

                #check for type differential of the labels inside an arc (see debug arc):
                vertex = Vertex(point_b, line_a, line_b)
                
                candidates.append(vertex)

            return candidates

        
        def build_barriers(surfaceTypes=None, labels=None, min_angle_threshold=np.radians(15), max_angle_threshold=np.radians(170)):
            for surface in self.room.get_surfaces(surfaceTypes=surfaceTypes, labels=labels):
                
                for poly in surface.polygons:
                    vertices = find_vertices(surface, poly, min_angle_threshold, max_angle_threshold)
                    self.barrier_candidates.append(SurfaceVertices(surface, vertices))
                
        
        #build_barriers([SurfaceType.Floor])
        build_barriers([SurfaceType.Wall])
        build_barriers(labels=box_like)

        return self.barrier_candidates

class RectangleFinder():
    def __init__(self, data, surface, vertices, hed):
        super().__init__()
        self.data = data
        self.room = data["room"]
        self.image = self.data["downscaled"]
        self.surface = surface
        self.vertices = vertices
        self.hed = hed

    def solve(self, max_iterations=3000, max_time=10.0, angle_threshold=np.radians(5), min_vp_confidence=0.01, min_hed_confidence=0.0):
        
        start_time = time.time() 
        if len(self.vertices) < 2 or len(self.surface.horizontal_vp) == 0 or len(self.surface.vertical_vp) == 0:
            return []

        horizontal_vp = self.surface.horizontal_vp[0]
        vertical_vp = self.surface.vertical_vp[0]

        candidates = []
        vp_mask = np.zeros(self.image.shape[:2], dtype=np.uint8)

        draw_vp(vp_mask, horizontal_vp, color=1, thickness=1)
        draw_vp(vp_mask, vertical_vp, color=1, thickness=1) 

        theta_thresh = np.cos(angle_threshold)

        points = []
        for poly in self.surface.polygons:
            for point in poly:
                points.append(point[0])

        for line in self.surface.lines:
            points.append(line.point_a)
            points.append(line.point_b)

        for line_data in horizontal_vp.inliers:
            points.append((line_data[0], line_data[1]))  

        for ransac_iter in range(max_iterations):
            if time.time() - start_time > max_time:
                break

            random.shuffle(points)

            point_a = points[0]
            point_b = points[1]

            line = Line(np.array([(point_a[0], point_a[1], point_b[0], point_b[1])], dtype=np.int).reshape(4))

            directions = np.array([line.direction]) 
            directions = directions / np.linalg.norm(directions, axis=1)[:, np.newaxis]
            locations = np.array([line.midpoint])

            vert_theta = angle_with_vp(vertical_vp.model, locations, directions)
            horiz_theta = angle_with_vp(horizontal_vp.model, locations, directions)
            
            is_vertical = vert_theta > theta_thresh
            is_horizontal = horiz_theta > theta_thresh

            if is_vertical or is_horizontal:
                num_samples = int(max(line.length / 5, 3))

                samples = LineFunctions.get_line_samples(line.point_a, line.point_b, vp_mask, num_samples)
                vp_confidence = sum(samples) / num_samples

                if vp_confidence < min_vp_confidence:
                    continue

                samples = LineFunctions.get_line_samples(line.point_a, line.point_b, self.hed, num_samples)
                hed_confidence = sum(samples) / num_samples

                if hed_confidence < min_hed_confidence:
                    continue

                candidates.append((line, is_horizontal))

        candidates.sort(key=lambda x:x[0].length * hed_confidence, reverse=True)

        return candidates

        
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

        self.candidates = BarrierFinder(self.data).solve()

        self.found_lines = []
        hed = cv2.resize(self.data["hed"], (self.image.shape[1], self.image.shape[0]))

        for candidate in self.candidates:
            rf = RectangleFinder(self.data, candidate.surface, candidate.vertices, hed)
            lines = rf.solve()
            self.found_lines.extend(lines)

        if im_logging_enabled(self.data):
            log_image(self.data, "barriers", self.get_debug_image())


    def get_debug_image(self):

        img_hsv = cv2.cvtColor(self.image, cv2.COLOR_RGB2HSV_FULL)
        hues = random.sample(range(0, 360), len(self.room.surfaces))

        #overlay probs
        for i in range(len(self.room.surfaces)):
            surface = self.room.surfaces[i]
            mask = surface.mask > 0

            max_value = 0.9
            if max_value > 0:
                img_hsv[:, :, 0][mask] = hues[i]
                img_hsv[:, :, 1][mask] = 255 * np.power(surface.probs[mask], 0.15)
                    
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB_FULL)

        #Line.draw_all(img, self.data["lines"], color=(80,80,80), thickness=2)

        # for candidate in self.candidates:

        #     color_a = (0, 0, 255) if candidate.surface.surfaceType.is_major else (0, 0, 180)
        #     color_b = (255, 255, 0) if candidate.surface.surfaceType.is_major else (180, 180, 0)

        #     for vertex in candidate.vertices:
        #         cv2.ellipse(img, vertex.center, (vertex.radius, vertex.radius), 0, np.degrees(vertex.line_a.angle), np.degrees(vertex.line_b.angle), (255, 0, 255), thickness=1) 
        #         #cv2.ellipse(img, vertex.center, (vertex.radius, vertex.radius), 0, np.degrees(vertex.line_b.angle), np.degrees(vertex.line_a.angle) + 360, [255, 0, 0], thickness=2) 

        #         cv2.line(img, vertex.line_a.point_a, vertex.line_a.point_b, color_a, thickness=2)
        #         cv2.line(img, vertex.line_b.point_a, vertex.line_b.point_b, color_b, thickness=2)
        
        # for candidate in self.candidates:
        #     color = (255, 0, 0) if candidate.surface.surfaceType.is_major else (255, 255, 255)
        #     thickness = 2 if candidate.surface.surfaceType.is_major else 1
        #     for vertex in candidate.vertices:
        #         cv2.drawMarker(img, vertex.center, color=color, thickness=thickness)

        
        for line, angle in self.found_lines:
            line.draw(img, color=(255,255,255), thickness=1)
            

        return img


