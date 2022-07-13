import cv2
import numpy as np
import numba as nb
import math
from scipy.spatial import distance
from numba.experimental import jitclass

from time import time
from scipy.spatial import distance

from pipeline.misc.utils import random_color
from pipeline.data.surface_type import SurfaceType
from pipeline.components.line import Line, merge_lines, draw_lines, line_on_image_edge
from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.logging import log_image, im_logging_enabled, LogLevel
from cambrian.LineFunctions import LineFunctions

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
        return ["lines", "hed_lines", "normals_lines"]
    
    def run(self, data):

        def log_lines(lines, name):
            if not im_logging_enabled(data, LogLevel.Lines): return

            debug = data["downscaled"].copy()
            thickness = max(int(math.hypot(debug.shape[0], debug.shape[1]) / 600), 1)
            draw_lines(debug, lines, thickness=thickness)
            log_image(data, name, debug)

        def line_cluster_image(lines):
            cluster_image = data["downscaled"].copy()

            draw_lines(cluster_image, lines, color=(0,0,0), thickness=1)

            max_cluster = max(line.cluster for line in lines)

            for i in range(0, max_cluster):
                cluster_lines = list(filter(lambda x: x.cluster==i, lines))
                if len(cluster_lines) > 0:
                    draw_lines(cluster_image, cluster_lines, color=random_color(), thickness=2)

            return cluster_image

        def find_lines(image, min_length, use_lsd=False, refine=cv2.LSD_REFINE_NONE, scale=1.0, sigma_scale=1.0, quant=2.0, ang_th=22.5, log_eps=0, density_th=0.7, n_bins=1024, canny_threshold=3):
            
            sx = data["downscaled"].shape[1] / image.shape[1]
            sy = data["downscaled"].shape[0] / image.shape[0]

            lines = list()

            if use_lsd:
                min_length_sq = (sx * min_length) ** 2
                lsd = cv2.createLineSegmentDetector(refine=refine, scale=scale, sigma_scale=sigma_scale, quant=quant, ang_th=ang_th, log_eps=log_eps, density_th=density_th, n_bins=n_bins)
                cv_lines = filter(lambda line: distance.sqeuclidean([line[0][0], line[0][1]], [line[0][2], line[0][3]]) >= min_length_sq and not line_on_image_edge((line[0][0], line[0][1]), (line[0][2], line[0][3]), image.shape[1], image.shape[0]), lsd.detect(image)[0])
            else:
                fld = cv2.ximgproc.createFastLineDetector(min_length, 1.41, 20, 240, canny_threshold, False)
                cv_lines = fld.detect(image)
            
            if cv_lines is not None:
                lines.extend(map(lambda line: Line(line[0][0] * sx, line[0][1] * sy, line[0][2] * sx, line[0][3] * sy), cv_lines))

            return lines

        bw = cv2.cvtColor(data["image"], cv2.COLOR_RGB2GRAY)

        self.height, self.width = bw.shape[:2]
        diagonal = np.hypot(self.width, self.height)


        operating_scale = 1500.0 / diagonal
        if operating_scale < 1.0:
            bw = cv2.resize(bw, (int(self.width * operating_scale), int(self.height * operating_scale)), cv2.INTER_CUBIC)
        
        self.height, self.width = bw.shape[:2]
        diagonal = np.hypot(self.width, self.height)

        min_length = int(diagonal / 50)
        print("min_length:", min_length)
        print("diagonal:", diagonal, self.width, self.height)

        lines = list()

        #find lines in BW image
        bw_lines_a = find_lines(bw, min_length)
        log_lines(bw_lines_a, "bw_lines_a")
        lines.extend(bw_lines_a)

        bw_lines_b = find_lines(bw, min_length, True, ang_th=5) #ang_th=22.5 was getting false positives
        log_lines(bw_lines_b, "bw_lines_b")
        lines.extend(bw_lines_b)

        bw_lines_c = find_lines(bw, 2*min_length, canny_threshold=7)
        
        if im_logging_enabled(data):
            merge_lines(bw_lines_c, search_width=max(diagonal/50, 3), angle_threshold=math.radians(1.0), remove_matches=False)
            log_image(data, "bw_lines_c_clusters", line_cluster_image(bw_lines_c))

        log_lines(bw_lines_c, "bw_lines_c")
        lines.extend(bw_lines_c)


        #find lines in hed hed edges
        hed_lines = find_lines(data["hed"], min_length, use_lsd=False, canny_threshold=3) #ang_th=22.5 was getting false positives
        hed_lines = merge_lines(hed_lines, search_width=diagonal/250, angle_threshold=math.radians(3))

        data["hed_lines"] = hed_lines

        log_lines(hed_lines, "hed_lines")

        #find lines in normals
        min_length = int(diagonal / 20)
        normals = np.uint8(data["normals"])
        normals = cv2.split(normals)
        normals_lines = []
        for i in range(0, 3):
            normals_lines.extend(find_lines(normals[i], min_length))

        data["normals_lines"] = normals_lines
        log_lines(normals_lines, "normals_lines")

        #merge all
        lines = merge_lines(lines, search_width=max(diagonal/500, 3), search_length=1.1, angle_threshold=math.radians(5.0))

        #assign clusters:
        merge_lines(lines, search_width=max(diagonal/150, 3), search_length=1.2, angle_threshold=math.radians(5.0), remove_matches=False)
        log_lines(lines, "merged_lines")

        if im_logging_enabled(data):
            cluster_image = line_cluster_image(lines)
            log_image(data, "line_clusters", cluster_image)

        data["lines"] = lines

