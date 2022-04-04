import cv2
import numpy as np
import math
from scipy.spatial import distance

from cambrian.frei_chen import frei_chen
from time import time

from pipeline.data.surface_type import SurfaceType
from pipeline.components.line import Line, merge_lines, draw_lines, line_on_image_edge
from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.logging import log_image, im_logging_enabled, LogLevel, Timer

def gabor(bw, theta, lambd, gamma = 0.0, psi = 0.0):
    ksize = lambd
    sigma = ksize * lambd
    result = cv2.filter2D(bw, cv2.CV_8UC1, cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F))
    return result

def sharpen(img, alpha=1.5, beta=-1.0, kernel_size = 21):
    smoothed = cv2.GaussianBlur(img, (kernel_size, kernel_size), kernel_size)
    return cv2.addWeighted(img, alpha, smoothed, beta, 0)

class PipelineLineFinder(PipelineStep):

    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.FindLines

    @property
    def required_keys(self) -> list:
        return ["image", "hed", "normals"]

    @property
    def output_keys(self) -> list:
        return ["lines"]
    
    def run(self, data):

        timer = Timer("line_finder")
        timer.disable()

#         7) FindLines
# line_finder.setup took 0.00 seconds
# line_finder.bw_lines_a took 0.02 seconds
# line_finder.bw_lines_b took 0.09 seconds
# line_finder.merge_lines BW took 5.96 seconds
# line_finder.bilateralFilter took 0.01 seconds
# line_finder.hed took 0.06 seconds
# line_finder.merge_lines HED took 0.76 seconds
# line_finder.normals_lines took 0.01 seconds
# line_finder.merge_lines normals_lines took 0.01 seconds
# line_finder.merge_lines final took 4.31 seconds
# 7) FindLines took 11.25 seconds

        bw = cv2.cvtColor(data["image"], cv2.COLOR_RGB2GRAY)

        self.height, self.width = bw.shape[:2]
        diagonal = np.hypot(self.width, self.height)

        operating_scale = 1500.0 / diagonal
        if operating_scale < 1.0:
            bw = cv2.resize(bw, (int(self.width * operating_scale), int(self.height * operating_scale)), cv2.INTER_CUBIC)

        timer.log_elapsed("setup")

        def log_lines(lines, name):
            if im_logging_enabled(data, LogLevel.Lines):
                debug = data["downscaled"].copy()
                thickness = max(int(math.hypot(debug.shape[0], debug.shape[1]) / 600), 1)
                draw_lines(debug, lines, thickness=thickness)
                log_image(data, name, debug)

        def find_lines(image, min_length, use_lsd=False, refine=cv2.LSD_REFINE_NONE, scale=1.0, sigma_scale=1.0, quant=2.0, ang_th=22.5, log_eps=0, density_th=0.7, n_bins=1024):
            min_length = int(min_length)
            if use_lsd:
                lsd = cv2.createLineSegmentDetector(refine=refine, scale=scale, sigma_scale=sigma_scale, quant=quant, ang_th=ang_th, log_eps=log_eps, density_th=density_th, n_bins=n_bins)
                lines = lsd.detect(image)[0]
                if lines is not None:
                    lines = list(filter(lambda line: distance.euclidean((line[0][0], line[0][1]), (line[0][2], line[0][3])) >= min_length, lines))
            else:
                fld = cv2.ximgproc.createFastLineDetector(min_length, 1.41, 200, 240, 3, False)
                lines = fld.detect(image)

            sx = data["downscaled"].shape[1] / image.shape[1]
            sy = data["downscaled"].shape[0] / image.shape[0]
            return list(map(lambda line: Line(line[0][0] * sx, line[0][1] * sy, line[0][2] * sx, line[0][3] * sy), lines)) if lines is not None else list()

        def extract_lines(poly, min_length):
            lines = []
            num_pts = len(poly)
            for i in range(num_pts):
                point_a = poly[i][0]
                point_b = poly[(i+1) % num_pts][0]
                if distance.euclidean(point_a, point_b) > min_length and not line_on_image_edge(point_a, point_b, data["downscaled"]):
                    lines.append(Line(point_a[0], point_a[1], point_b[0], point_b[1]))

            return lines

        min_length = int(diagonal / 100)

        lines = []

        #pull lines from semantic contours.
        for surfaceType in SurfaceType:
            mask = data["isolated"][surfaceType]
            isolated = np.zeros(mask.shape, dtype=np.uint8)
            isolated[mask > 0.9] = 1

            mask_bordered = cv2.copyMakeBorder(isolated, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0) 
            contours, _ = cv2.findContours(mask_bordered, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            epsilon = 7

            for contour in contours:
                shape = contour.shape
                contour = (contour.flatten() - 1).reshape(shape)
                poly = cv2.approxPolyDP(contour, epsilon, True)
                lines.extend(extract_lines(poly, min_length))

        #find lines in BW image
        bw_lines_a = find_lines(bw, min_length)
        lines.extend(bw_lines_a)

        timer.log_elapsed("bw_lines_a")

        bw_lines_b = find_lines(bw, min_length, True, ang_th=17) #ang_th=22.5 was getting false positives
        lines.extend(bw_lines_b)

        timer.log_elapsed("bw_lines_b")

        edges = (frei_chen(bw) * 3.0 * 255.0).astype(np.uint8)
        edges_lines = find_lines(edges, min_length * 2.0, True)
        #edges_lines = merge_lines(edges_lines, search_width=diagonal/400, angle_threshold=math.radians(3))

        log_lines(edges_lines, "edges_lines")

        lines.extend(edges_lines)

        lines = merge_lines(lines, search_length=1.0, search_width=diagonal/800, angle_threshold=math.radians(3))

        timer.log_elapsed("merge_lines BW")

        #find lines in hed hed edges
        sx = data["downscaled"].shape[1] / data["hed"].shape[1]
        sy = data["downscaled"].shape[0] / data["hed"].shape[0]

        hed_lines = find_lines(data["hed"], min_length, use_lsd=True, ang_th=12) #ang_th=22.5 was getting false positives
        timer.log_elapsed("hed")

        hed_lines = merge_lines(hed_lines, search_length=0.5, search_width=diagonal/200, angle_threshold=math.radians(3))

        timer.log_elapsed("merge_lines HED")

        if len(hed_lines) > 0: 
            #log_lines(hed_lines, "hed_lines")
            lines.extend(hed_lines)

        #find lines in normals
        min_length = int(diagonal / 20)
        normals = np.uint8(data["normals"])
        #log_image(data, "normals", normals)
        normals = cv2.split(normals)
        normals_lines = []
        for i in range(0, 3):
            normals_lines.extend(find_lines(normals[i], min_length))

        timer.log_elapsed("normals_lines")
        
        if len(normals_lines) > 0:
            #cleanup normals
            normals_lines = merge_lines(normals_lines, search_width=diagonal/300)
            timer.log_elapsed("merge_lines normals_lines")

            lines.extend(normals_lines)

        #merge all
        lines = merge_lines(lines, search_width=min(diagonal/300, 8))
        timer.log_elapsed("merge_lines final")

        # print("8. elapsed %.2f" % (time() - start)); start = time()

        log_lines(lines, "merged_lines")

        data["lines"] = lines

