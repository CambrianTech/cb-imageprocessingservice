import cv2
import numpy as np
import math

from collections.abc import Sequence
from bisect import bisect_left, bisect_right

from cambrian.frei_chen import frei_chen

from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.logging import log_image, im_logging_enabled, LogLevel
from pipeline.Line import Line, bounding_box, draw_line, sqeuclidean, merge

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

        bw = cv2.cvtColor(data["image"], cv2.COLOR_RGB2GRAY)

        self.height, self.width = bw.shape[:2]
        diagonal = np.hypot(self.width, self.height)

        def log_lines(lines, name):
            if im_logging_enabled(data, LogLevel.Lines):
                debug = data["image"].copy()
                thickness = max(int(math.hypot(debug.shape[0], debug.shape[1]) / 600), 1)
                [draw_line(line, debug,  thickness=thickness) for line in lines]
                log_image(data, name, debug)

        #transform_result = lambda x: list(map(lambda x: Line(x.reshape(4), sx, sy), result))
        transform_result = lambda result: list(map(lambda x: Line(int(x[0][0] * sx), int(x[0][1] * sy), int(x[0][2] * sx), int(x[0][3] * sy)), result))

        fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 60.0), 1.41, 200, 240, 3, False)
        lines = []

        #find lines in BW image
        sx = data["image"].shape[1] / bw.shape[1]
        sy = data["image"].shape[0] / bw.shape[0]
        result = fld.detect(bw)
        if result is not None and len(result) > 0: 
            lines.extend(transform_result(result))
        log_lines(lines, "bw_lines")


        #find lines in hed hed edges
        sx = data["image"].shape[1] / data["hed"].shape[1]
        sy = data["image"].shape[0] / data["hed"].shape[0]

        result = fld.detect(data["hed"])
        if result is not None and len(result) > 0: 
            #lines are made parallel by thickness of source image edges
            hed_lines = merge(transform_result(result), search_width=diagonal/100)
            log_lines(hed_lines, "hed_lines")
            lines.extend(hed_lines)

        #find lines in normals
        fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 20.0), 1.41, 200, 240, 3, False)
        normals = np.uint8(data["normals"])
        #log_image(data, "normals", normals)
        sx = data["image"].shape[1] / normals.shape[1]
        sy = data["image"].shape[0] / normals.shape[0]
        normals = cv2.split(normals)
        normals_lines = []
        for i in range(0, 3):
            result = fld.detect(normals[i])
            if result is not None and len(result) > 0:
                normals_lines.extend(transform_result(result))
        
        if len(normals_lines) > 0:
            #cleanup normals
            normals_lines = merge(normals_lines, search_width=diagonal/300)
            log_lines(normals_lines, "normals_lines")
            lines.extend(normals_lines)

        #find lines in gabor edges:
        gabor_scale = 1500.0 / diagonal
        bw_res = cv2.resize(bw, (int(self.width * gabor_scale), int(self.height * gabor_scale)), cv2.INTER_CUBIC) if gabor_scale < 1.0 else bw
        v_gabor = gabor(bw_res, 0, 7)
        h_gabor = gabor(bw_res, np.pi/2.0, 9)
        edges = cv2.addWeighted(v_gabor, 3.0, h_gabor, 3.0, -20)
        edges = cv2.bilateralFilter(edges, 5, 5, 5)
        edges = cv2.resize(edges, (self.width, self.height), interpolation = cv2.INTER_CUBIC)

        sx = data["image"].shape[1] / edges.shape[1]
        sy = data["image"].shape[0] / edges.shape[0]
        result = fld.detect(edges)
        if result is not None and len(result) > 0: 
            gabor_lines = merge(transform_result(result), search_width=diagonal/100)
            log_lines(gabor_lines, "gabor_lines")
            lines.extend(gabor_lines)

        #frei chen edges:
        clean_edges = frei_chen(bw)
        result = fld.detect(bw - (clean_edges * 5.0).astype("uint8"))
        if result is not None and len(result) > 0: 
            frei_lines = transform_result(result)
            log_lines(frei_lines, "frei_lines")
            lines.extend(frei_lines)


        #merge all
        lines = merge(lines, search_width=diagonal/200)

        log_lines(lines, "merged_lines")

        data["lines"] = lines
