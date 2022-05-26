import cv2
import numpy as np
import numba as nb
import math
from scipy.spatial import distance

from cambrian.frei_chen import frei_chen
from time import time
from scipy.spatial import distance

from pipeline.components.base_process import BaseProcess, Multiprocessor
from pipeline.data.surface_type import SurfaceType
from pipeline.components.line import Line, merge_lines, draw_lines, line_on_image_edge
from pipeline.core import PipelineStep, PipelineStepIndex
from pipeline.data.logging import log_image, im_logging_enabled, LogLevel, Timer

class LineFinderProcess(BaseProcess):
    def find_lines(self, arg1, arg2):
        print(arg1, arg2)

        return 10

class PipelineLineFinder(PipelineStep):

    def __init__(self, pipeline):
        super().__init__(pipeline)

        self.mp = Multiprocessor(LineFinderProcess, num_workers=2)
        self.mp.start()

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

        def log_lines(lines, name):
            if not im_logging_enabled(data, LogLevel.Lines): return

            debug = data["downscaled"].copy()
            thickness = max(int(math.hypot(debug.shape[0], debug.shape[1]) / 600), 1)
            draw_lines(debug, lines, thickness=thickness)
            log_image(data, name, debug)

        def find_lines(image, min_length, use_lsd=False, refine=cv2.LSD_REFINE_NONE, scale=1.0, sigma_scale=1.0, quant=2.0, ang_th=22.5, log_eps=0, density_th=0.7, n_bins=1024):
            
            sx = data["downscaled"].shape[1] / image.shape[1]
            sy = data["downscaled"].shape[0] / image.shape[0]

            lines = list()

            if use_lsd:
                min_length_sq = (sx * min_length) ** 2
                lsd = cv2.createLineSegmentDetector(refine=refine, scale=scale, sigma_scale=sigma_scale, quant=quant, ang_th=ang_th, log_eps=log_eps, density_th=density_th, n_bins=n_bins)
                cv_lines = filter(lambda line: distance.sqeuclidean([line[0][0], line[0][1]], [line[0][2], line[0][3]]) >= min_length_sq, lsd.detect(image)[0])
            else:
                fld = cv2.ximgproc.createFastLineDetector(min_length, 1.41, 200, 240, 3, False)
                cv_lines = fld.detect(image)

            if cv_lines is not None:
                lines.extend(map(lambda line: Line(line[0][0] * sx, line[0][1] * sy, line[0][2] * sx, line[0][3] * sy), cv_lines))

            return lines

        timer = Timer("line_finder")
        timer.disable()

        bw = cv2.cvtColor(data["image"], cv2.COLOR_RGB2GRAY)

        self.height, self.width = bw.shape[:2]
        diagonal = np.hypot(self.width, self.height)

        operating_scale = 1500.0 / diagonal
        if operating_scale < 1.0:
            bw = cv2.resize(bw, (int(self.width * operating_scale), int(self.height * operating_scale)), cv2.INTER_CUBIC)

        timer.log_elapsed("setup")

        min_length = int(diagonal / 50)

        lines = list()
        
        self.mp.schedule('find_lines', 'hello', 'world')
        self.mp.await_completion()

        #find lines in BW image
        bw_lines_a = find_lines(bw, min_length)

        lines.extend(bw_lines_a)

        timer.log_elapsed("bw_lines_a")

        bw_lines_b = find_lines(bw, min_length, True, ang_th=17) #ang_th=22.5 was getting false positives
        lines.extend(bw_lines_b)

        timer.log_elapsed("bw_lines_b")

        # edges = (frei_chen(bw) * 3.0 * 255.0).astype(np.uint8)
        # edges_lines = find_lines(edges, min_length * 2.0, True, ang_th=17)
        #edges_lines = merge_lines(edges_lines, search_width=diagonal/400, angle_threshold=math.radians(3))
        #log_lines(edges_lines, "edges_lines")
        #lines.extend(edges_lines)

        lines = merge_lines(lines, search_length=1.0, search_width=diagonal/800, angle_threshold=math.radians(3))

        timer.log_elapsed("merge_lines BW")

        #find lines in hed hed edges
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

