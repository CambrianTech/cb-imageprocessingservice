import numpy as np
from scipy import ndimage
import cv2
from skimage.morphology import remove_small_objects

from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.misc.utils import random_color
from pipeline.data.logging import log_image, log_mask, log_segmentation_image, im_logging_enabled, Timer
from pipeline.components.scene import Scene
from pipeline.components.surface import Surface
from pipeline.components.line import draw_lines


class PipelineSurfaceMerger(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.SurfaceMerger

    @property
    def required_keys(self) -> list:
        return ["planes", "downscaled", "isolated", "dimensions"]

    @property
    def output_keys(self) -> list:
        return ["room"]

    def run(self, data):

        print("MERGE walls")

        debug = data["downscaled"].copy()

        lines_mask = np.zeros(data["downscaled"].shape[:2], dtype="uint8")
        draw_lines(lines_mask, data["lines"], color=255, thickness=1)

        draw_lines(debug, data["lines"], color=(0,0,255), thickness=1)

        for surface in data["room"].get_surfaces([SurfaceType.Wall]):

            if surface.destroyed: continue
            
            # log_mask(data, surface.name + "_mask", surface.mask)
            # log_mask(data, surface.name + "_mask_edges", surface.mask_edges)
            # log_mask(data, surface.name + "_mask_expanded", surface.mask_expanded)

            smaller_neighbors = list(filter(lambda s: s.surfaceType == surface.surfaceType and s.max_area < surface.max_area and not s.destroyed, surface.neighbors))

            if len(smaller_neighbors) == 0: continue

            print("surface %s has smaller neighbors" % surface.name, [neighbor.name for neighbor in smaller_neighbors])

            for candidate in smaller_neighbors:

                contours = surface.intersection(candidate) 

                mask = np.zeros(data["downscaled"].shape[:2], dtype="uint8")
                #cv2.drawContours(mask, contours, -1, 255, -1)

                color = random_color()

                for contour in contours:
                    moments = cv2.moments(contour)

                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])

                    cv2.circle(debug, (cX, cY), 10, color, 2)

                    cv2.circle(mask, (cX, cY), 10, 255, -1)


                mask = cv2.bitwise_and(mask, lines_mask)

                if cv2.countNonZero(mask) < 5:
                    #surface.merge(candidate)
                    cv2.drawContours(debug, contours, -1, color, -1)
                else:
                    cv2.drawContours(debug, contours, -1, color, 2)
                
                    

        log_image(data, "room_remerged", data["room"].get_debug_image())

        log_image(data, "intersections", debug)
