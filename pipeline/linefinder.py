import cv2
import numpy as np
import math
from scipy.spatial import distance
from time import time

from .core import PipelineStep
from .line import Line

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
    def required_keys(self) -> list:
        return ["image", "hed", "normals"]

    @property
    def output_keys(self) -> list:
        return ["lines"]
    
    def run(self, data):

        bw = cv2.cvtColor(data["image"], cv2.COLOR_RGB2GRAY)

        self.height, self.width = bw.shape[:2]
        diagonal = np.hypot(self.width, self.height)

        start = time()

        operating_scale = 1500.0 / diagonal
        if operating_scale < 1.0:
            bw = cv2.resize(bw, (int(self.width * operating_scale), int(self.height * operating_scale)), cv2.INTER_CUBIC)

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
            return list(map(lambda x: Line(x.reshape(4), sx, sy), lines)) if lines is not None else list()

        min_length = int(diagonal / 80)

        lines = []

        #find lines in BW image
        bw_lines_a = find_lines(bw, min_length)
        if len(bw_lines_a) > 0:
            lines.extend(bw_lines_a)

        bw_lines_b = find_lines(bw, min_length, True, ang_th=17) #ang_th=22.5 was getting false positives
        if len(bw_lines_b) > 0:
            lines.extend(bw_lines_b)

        lines = Line.merge(lines, search_length=1.0, search_width=diagonal/800, angle_threshold=math.radians(3))

        #find lines in hed hed edges
        sx = data["downscaled"].shape[1] / data["hed"].shape[1]
        sy = data["downscaled"].shape[0] / data["hed"].shape[0]

        hed = data["hed"].copy()
        hed = cv2.bilateralFilter(hed, 13, 40, 9) #todo: apply non-maxima-suppression (NMS) to image instead
        hed_lines = find_lines(hed, min_length, use_lsd=True, ang_th=12) #ang_th=22.5 was getting false positives

        if len(hed_lines) > 0: 
            hed_lines = Line.merge(hed_lines, search_length=0.5, search_width=diagonal/200, angle_threshold=math.radians(3))
            lines.extend(hed_lines)

        #find lines in normals
        min_length = int(diagonal / 20)
        normals = np.uint8(data["normals"])
        #log_image(data, "normals", normals)
        normals = cv2.split(normals)
        normals_lines = []
        for i in range(0, 3):
            new_lines = find_lines(normals[i], min_length)
            if len(new_lines) > 0:
                normals_lines.extend(new_lines)
        
        if len(normals_lines) > 0:
            #cleanup normals
            normals_lines = Line.merge(normals_lines, search_width=diagonal/300)
            lines.extend(normals_lines)

        #merge all
        lines = Line.merge(lines, search_width=min(diagonal/400, 8))

        print("PipelineLineFinder found %d lines" % len(lines))

        data["lines"] = lines

