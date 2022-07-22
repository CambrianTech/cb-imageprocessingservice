from email.errors import BoundaryError
from re import M
from cv2 import threshold
from matplotlib import lines
import numpy as np
from scipy import ndimage
import cv2
from termcolor import colored
from skimage.morphology import remove_small_holes
from skimage.segmentation import watershed
from scipy.spatial import distance

from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.misc.utils import resize_array, random_color, overlay_mask, sample_at_point, scale_contour, color_to_normal
from .planegeometry import Dimension
from pipeline.data.logging import log_image, log_segmentation_image, im_logging_enabled, log_markers, Timer
from pipeline.components.line import Line, draw_lines, line_on_image_edge, merge_lines
from pipeline.components.surface import Surface, SurfaceBarrier
from pipeline.data.ade20k import ADE20K
from pipeline.stages.vanishingpointfinder import get_votes, get_contour_lines

class SurfaceSolver():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.room = self.data["room"]

    def solve(self, ignore_rugs=False):
        
        #add all the applicable surfaces:
        timer = Timer("room")

        if ignore_rugs:
            self.room.isolated_labels[self.room.isolated_labels == SurfaceType.OnFloor.value] = SurfaceType.Floor.value

        #prepare
        self.lines_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)
        draw_lines(self.lines_mask, self.data["semantic_lines"], color=255, thickness=1, lineType=cv2.LINE_4)

        self.height, self.width = self.room.image.shape[:2]
        self.area = self.height * self.width
        self.diagonal = np.hypot(self.width, self.height)
        self.refine_surfaces()

        #preserve plane context information i.e. probs < min_confidence are ignored
        if im_logging_enabled(self.data):
            log_image(self.data, "room_refined", self.room.get_debug_image())

        self.add_missing_surfaces(self.lines_mask)
        self.room.refresh_surfaces()

        if im_logging_enabled(self.data):
            log_image(self.data, "room_missing_added", self.room.get_debug_image())

        self.build_fan()
        
        if im_logging_enabled(self.data):
            log_image(self.data, "room_fan", self.room.get_debug_image())


        self.find_trim()

        if im_logging_enabled(self.data):
            self.debug_vps()

        self.merge_fan()
        self.room.refresh_surfaces()

        log_image(self.data, "room_merged", self.room.get_debug_image())

        self.surface_cleanup()
        self.room.refresh_surfaces()

        if im_logging_enabled(self.data):
            log_image(self.data, "room_solved", self.room.get_debug_image(hires=True))

        #self.surface_vp_matching()

        timer.log_all_events()

        return self.room

    def surface_cleanup(self):
        #remove leftover bad ones. Merge in?
        min_area = 1/300
        total_area = self.room.image.shape[0] * self.room.image.shape[1]
        area_threshold = int(total_area * min_area)

        for surface in self.room.surfaces:
            if surface.bestLabel is None or (surface.max_area < area_threshold and surface.surfaceType == SurfaceType.Wall):
                neighbors = list(filter(lambda s: s.surfaceType==surface.surfaceType and not s.destroyed and s.bestLabel is not None, surface.neighbors))
                if len(neighbors) > 0:
                    best = sorted(neighbors, key=lambda s: s.max_area, reverse=True)
                    best[0].merge(surface)
                else:
                    candidates = list(filter(lambda s:s != surface, self.room.get_surfaces([surface.surfaceType])))
                    best = sorted(candidates, key=lambda s: sum(abs(s.normals_mean - surface.normals_mean)))
                    best[0].merge(surface)



    def find_trim(self):

        candidate_surfaces = []
        debug = self.room.image.copy()

        min_area = 1/100
        total_area = self.room.image.shape[0] * self.room.image.shape[1]
        area_threshold = int(total_area * min_area)

        def is_mask_line(line, mask, num_points=7, num_matches=3):
            line_points = np.linspace(line.point_a, line.point_b, num_points)
            count = 0
            for point in line_points:
                if point[0] < 0 or point[0] >= mask.shape[1] or point[1] < 0 or point[1] >= mask.shape[0]: continue

                if mask[int(point[1]), int(point[0])] > 0:
                    count += 1

                if count > num_matches:
                    return True

            return False

        trim_surfaces = self.room.get_surfaces([SurfaceType.Wall, SurfaceType.OnWall, SurfaceType.Ceiling])
        trim_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)

        for surface in trim_surfaces:
            trim_mask[surface.mask_edges > 0] = 1

        debug[trim_mask > 0] = [0,0,255]
        alpha = 0.3
        debug = cv2.addWeighted(debug, alpha, self.room.image, 1 - alpha, 0)

        for surface in trim_surfaces:

            for contour in surface.contours:
                if cv2.contourArea(contour) <= area_threshold and surface.surfaceType in [SurfaceType.Wall, SurfaceType.Ceiling]:
                    cv2.drawContours(debug, [contour], 0, random_color(), -1)            

        line_candidates = list(filter(lambda line: is_mask_line(line, trim_mask), self.data["vp_lines"]))

        draw_lines(debug, self.data["vp_lines"], color=(255, 0, 0), thickness=1,lineType=cv2.LINE_AA)
        draw_lines(debug, line_candidates, color=(255, 255, 0), thickness=2,lineType=cv2.LINE_AA)

        self.room.barrier_lines = line_candidates

        log_image(self.data, "trim_candidates", debug)


    def debug_vps(self):
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for c in range(1000): colors.append(random_color())

        for surface in self.room.get_surfaces([SurfaceType.Wall]):

            if len(surface.horizontal_vps) == 0: continue

            debug = self.data["downscaled"].copy()
            debug[surface.mask_expanded > 0] = random_color()

            #draw_lines(debug, surface.lines, color=(50, 50, 50), thickness=2,lineType=cv2.LINE_AA)

            for vp in surface.horizontal_vps:
                index = self.room.horizontal_vps.index(vp) + 1
                color = colors[index]
                draw_lines(debug, vp.inliers, color=(color[0], color[1], color[2]), thickness=2, lineType=cv2.LINE_AA)

            log_image(self.data, "vanishing_pts_%s" % surface.name, debug)

    def build_fan(self):

        vertical_vp = self.room.vertical_vp
        vertical_lines = vertical_vp.inliers.copy()

        vertical_labels = self.room.isolated_labels.copy()

        # # vertical_labels[self.lines_mask > 0] = np.amax(vertical_labels) + 1

        surface_image = self.room.image.copy()
        surface_probs = np.zeros_like(self.room.image)

        # for surfaceType in [SurfaceType.Wall, SurfaceType.OnWall, SurfaceType.Floor, SurfaceType.Ceiling, SurfaceType.OnFloor,SurfaceType.OnCeiling,SurfaceType.Other]:
        for surfaceType in [SurfaceType.Wall, SurfaceType.OnWall]:
        # for surfaceType in [SurfaceType.Wall]:
            surfaces_of_type = self.room.get_surfaces([surfaceType])
            refined_type = np.sum([surface.mask for surface in surfaces_of_type],0)

            mask = np.uint8(refined_type>0)

            if len(mask.shape) != 2:
                continue

            type_surface = Surface(self.data)
            type_surface.mask = mask

            type_lines = type_surface.lines

            for surface in surfaces_of_type:
                #print(np.unique(surface.probs))

                surface_mask = surface.mask
                mask_coor = np.nonzero(surface_mask)
                    
                mask_coor_vp_x = mask_coor[1] - vertical_vp.model[:2][0]
                mask_coor_vp_y = mask_coor[0] - vertical_vp.model[:2][1]

                if len(mask_coor_vp_x) == 0 or len(mask_coor_vp_y) == 0:
                    continue

                # inefficient
                mask_coor_vp_angles = np.arctan2(mask_coor_vp_y, mask_coor_vp_x) % np.pi

                surface_lines = type_lines.copy()

                surface_lines = list(filter(lambda x: x in vertical_lines, surface_lines))
                if len(surface_lines) == 0:
                    continue

                surface_lines = list(map(lambda line: line.extended(.5, vanishing_point=vertical_vp.model[:2]), surface_lines))
                         
                surface_lines.sort(key=lambda x:x.angle%np.pi)

                boundary_lines = []

                current_line = surface_lines[0]
                boundary_lines.append(current_line)

                for i in range(1, len(surface_lines)-1):
                    line = surface_lines[i]
                    # print("cluster", i,line.cluster)
                    if line.cluster != current_line.cluster:
                        # print("adding", i-1,i)
                        boundary_lines.append(current_line)
                        boundary_lines.append(line)
                    
                    current_line = line

                boundary_lines.append(surface_lines[-1])
                
                surface_lines = boundary_lines
                surface_angles = [line.angle%np.pi for line in surface_lines]
                pixel_angle_width = abs((np.amax(mask_coor_vp_angles) - np.amin(mask_coor_vp_angles))/(np.amax(mask_coor_vp_x)-np.amin(mask_coor_vp_x)))
                #print("pixel_angle_width", np.degrees(pixel_angle_width))
                surface_angles = [surface_angle for surface_angle in surface_angles if surface_angle >= np.amin(mask_coor_vp_angles) and surface_angle <= np.amax(mask_coor_vp_angles)]
                surface_angles = np.insert(surface_angles,0,np.amin(mask_coor_vp_angles)- 2*pixel_angle_width)
                surface_angles = np.append(surface_angles,np.amax(mask_coor_vp_angles)+ 2*pixel_angle_width)
                surface_angles.sort()
                # surface_angles = np.insert(surface_angles,0,surface_angles[0] + np.sign(surface_angles[0]-surface_angles[-1])* np.pi/4.)
                # surface_angles = np.append(surface_angles,surface_angles[-1] - np.sign(surface_angles[0]-surface_angles[-1])* np.pi/4.)
                #print(surface.name)
                #print(surface_angles)
                

                # surface_angles = mask_coor_vp_angles[surface_mask[mask_coor]>0]
                # max_surface_angle = np.amax(surface_angles)
                # min_surface_angle = np.amin(surface_angles)

         
                # surface_image[surface_mask>0] = color
                #print(surface.name, np.unique(surface.probs))

                surface_color = random_color()
                for i in range(len(surface_angles)-1):   
                    # print(surface_angles[i], surface_angles[i+1])

                    good = np.logical_and(mask_coor_vp_angles >= surface_angles[i], mask_coor_vp_angles <= surface_angles[i+1]) 
                    slice = mask_coor[0][good], mask_coor[1][good]

                    if  np.count_nonzero(good) > 10:
                        
                        index = np.amax(vertical_labels) + 1
                        slice_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)
                        slice_mask[slice] = 255

                        contours, hierarchy = cv2.findContours(slice_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                        sliver_min = 10000
                        sliver_max = 0

                        for contour in contours:
                            rect = cv2.minAreaRect(contour)

                            sliver_min = min(rect[1][0], rect[1][1], sliver_min)
                            sliver_max = max(rect[1][0], rect[1][1], sliver_max)

                        if sliver_max == 0: continue

                        thinness = sliver_min / sliver_max

                        #print("thinness", thinness)
                        # exit()

                        angle_diff = abs(surface_angles[i+1]-surface_angles[i])
                        #print(i, np.degrees(angle_diff))

                        if thinness > 0.15 and angle_diff > 2 * pixel_angle_width:
                    
                            prob = np.mean(surface.probs[slice_mask>0])

                            normal =  np.mean(self.data["room"].normals[slice_mask > 0], axis=0)
                            surface_probs[slice_mask>0] = [surface_color[0]*prob, surface_color[1]*prob, surface_color[2]*prob]
                            # surface_probs[slice_mask>0] = prob

                            if prob<.1:
                                surface.mask[slice_mask>0]=0
                                vertical_labels[slice] = np.amax(vertical_labels) + 1

                                # slice_surface = Surface(self.data, None, surfaceType)
                                slice_surface = surface.clone()
                                slice_surface.mask = slice_mask

                                if len(slice_surface.contours):
                                    self.room.add_surface(slice_surface)
                                    slice_surface.bestLabel

                                    surface_image[slice_mask>0] = random_color()

                                    surface_image[slice_mask>0] = normal

                            else:
                                surface_image[slice_mask>0] = surface_color
                        else:
                            surface_angles[i+1] = surface_angles[i]
                            # surface.mask[slice_mask>0]=0
                            
                            # surface_image[surface.mask>0] = np.mean(self.data["room"].normals[surface.mask > 0], axis=0)
                    else:
                        surface_angles[i+1] = surface_angles[i]
                        # surface.mask[slice]=0


                surface.mask_changed()
      
                # draw_lines(surface_probs, surface.lines, color=random_color(), thickness=2, lineType=cv2.LINE_AA)
                cv2.drawContours(surface_probs, surface.contours, -1, random_color(), -1)
                # draw_lines(surface_image, surface.lines, color=random_color(), thickness=2, lineType=cv2.LINE_AA)

        # # vertical_labels[self.lines_mask > 0] = np.amax(vertical_labels) + 1

        log_image(self.data, "room_surface_image", surface_image)
        log_image(self.data, "room_surface_probs", surface_probs)

        self.room.refresh_surfaces()

    def merge_fan(self):

        def should_merge(surface_a, surface_b):

            if surface_a.bestLabel != surface_b.bestLabel:
                return False

            if surface_a.surfaceType in [SurfaceType.OnWall, SurfaceType.Other, SurfaceType.OnCeiling] and surface_b in surface_a.neighbors:
                return True

            if surface_b.surfaceType in [SurfaceType.Floor, SurfaceType.OnFloor, SurfaceType.Ceiling]:
                return True

            vps_a = surface_a.horizontal_vps
            vps_b = surface_b.horizontal_vps

            print("Comparing %s to %s" % (surface_a.name, surface_b.name))

            if len(vps_a) > 0 and len(vps_b) > 0:
                vps_intersection = list(set(vps_a) & set(vps_b))
                if len(vps_intersection) == 0:
                    #print("Cannot merge surface %s with %s" % (surface_a.name, surface_b.name), vps_a, vps_b)
                    return False
            else:
                vps_intersection = None

            surface_a_normal = color_to_normal(surface_a.normals_mean)
            surface_b_normal = color_to_normal(surface_b.normals_mean)

            cos_normal = np.dot(surface_a_normal, surface_b_normal)
            angle = np.arccos(cos_normal)

            if angle <= np.radians(15) and (vps_intersection is None or len(vps_intersection) > 0):
                print("Merge %s with %s due to 15 degree normals" % (surface_a.name, surface_b.name), surface_a_normal, surface_b_normal)
                return True
            elif angle <= np.radians(45):

                if vps_intersection is not None and len(vps_intersection) == 1:
                    #print("Merge %s with %s due to matching vanishing points" % (surface_a.name, surface_b.name))
                    return True
        
                #see if there's a line through the intersection
                contours = surface_a.intersection(surface_b) 

                if contours is None:
                    print("Cannot merge %s with %s due to no line between them" % (surface_a.name, surface_b.name))
                    return False

                mask = np.zeros(self.data["downscaled"].shape[:2], dtype="uint8")

                for contour in contours:
                    moments = cv2.moments(contour)

                    if moments["m00"] == 0: continue

                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])

                    cv2.circle(mask, (cX, cY), 10, 255, -1)

                mask = cv2.bitwise_and(mask, self.lines_mask)

                if cv2.countNonZero(mask) < 5:
                    print("Merge %s with %s after finding line between them" % (surface_a.name, surface_b.name))
                    return True

            # thickness = surface_b.bounds[1][0] / surface_b.bounds[1][1]
            # thickness = min(thickness, 1.0/thickness)

            # if thickness < 0.1: 
            #     return True

            print("No match for %s with %s" % (surface_a.name, surface_b.name), angle)

            return False

        print(colored("\nPerforming sweep merge", attrs=['bold']))

        for surfaceType in SurfaceType:

            surfaces_of_type = self.room.get_surfaces([surfaceType])

            for current_surface in surfaces_of_type:

                if current_surface.destroyed: continue

                if surfaceType in [SurfaceType.Wall, SurfaceType.OnWall]:
                    candidates = filter(lambda s: s.surfaceType == current_surface.surfaceType, current_surface.neighbors)
                else:
                    candidates = surfaces_of_type

                for candidate in candidates:

                    if candidate == current_surface or candidate.destroyed: continue

                    if should_merge(current_surface, candidate):
                        current_surface.merge(candidate)

                if surfaceType == SurfaceType.Wall:
                    neighbors = list(filter(lambda s:s.surfaceType == current_surface.surfaceType and not s.destroyed, current_surface.neighbors))
                    print("surface %s" % current_surface.name, [neighbor.name for neighbor in neighbors])
        
        self.room.refresh_surfaces()

        print(colored("Sweep merge completed\n", attrs=['bold']))

    def surface_vp_matching(self):
        surfaces = []
        #vertical_labels[vertical_labels>=0] = -1
        normals_accumulated = [0,0,0]

        for surfaceType in SurfaceType: 
            surfaces.extend(self.data["room"].get_surfaces([surfaceType]))

        cluster_index = 0
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for c in range(100):
            colors.append(random_color())

        if len(self.room.horizontal_vps) > 0:
            vps = self.room.horizontal_vps
            vps = np.insert(vps, 0, self.room.vertical_vp)
        else:
            vps = [self.room.vertical_vp]

        for surface in surfaces:
            surface_img = self.room.image.copy()
            if surface.surfaceType == SurfaceType.Other:
                continue

            lines = surface.lines
            surface_img[surface.mask>0] = random_color()
            lines.extend(get_contour_lines(self.room.image,surface,4,use_contours=True))

            if len(lines) == 0:
                continue    

            max_score_1 = 0
            max_score_2 = 0
            best_lines_1 = None
            best_lines_2 = None
            vp_index_1 = None
            vp_index_2 = None

            surface.barriers = []

            for i in range(len(vps)):

                votes = get_votes(lines,vps[i].model,np.radians(2.0))

                score = sum(votes)

                if surface.surfaceType == SurfaceType.Wall or surface.surfaceType == SurfaceType.OnWall:
                    if best_lines_1 is None:
                        max_score_1 = 100000

                        best_lines_1 = [lines[j] for j in range(len(lines)) if votes[j] > 0]
                        vp_index_1 = 0
                    if score > max_score_2 and i!=0:
                        max_score_2 = score
                        vp_index_2 = i
                        best_lines_2 = [lines[j] for j in range(len(lines)) if votes[j] > 0]
                elif i>0:
                    if score > max_score_1:
                        max_score_2 = max_score_1
                        max_score_1 = score
                        vp_index_2 = vp_index_1
                        vp_index_1 = i
                        best_lines_2 = best_lines_1
                        best_lines_1 = [lines[j] for j in range(len(lines)) if votes[j] > 0]
                    elif score > max_score_2:
                        max_score_2 = score
                        best_lines_2 = [lines[j] for j in range(len(lines)) if votes[j] > 0]
                        vp_index_2 = i

                #print("Surface %s has %f votes (max scores %f, %f) from vps %i:" % (surface.name, score,max_score_1, max_score_2, i))
            
            #if vp_index_1 is not None and vp_index_2 is not None:
            #    print("vp_index_1: %i, vp_index_2: %i" % (vp_index_1, vp_index_2))

            mask_coor = np.nonzero(surface.mask>0)

            if best_lines_1 is not None :
                color2 = colors[vp_index_1]
                vp_pt = vps[vp_index_1].model
                vp_pt = vp_pt[:2]/vp_pt[2]
                mask_coor_vp_x = mask_coor[1] - vp_pt[0]
                mask_coor_vp_y = mask_coor[0] - vp_pt[1]
                mask_coor_vp_angles = np.arctan2(mask_coor_vp_y, mask_coor_vp_x)
                surface_angles = [line.angle for line in best_lines_1]

                if len(surface_angles) == 0:
                    continue

                surface_angles = np.insert(surface_angles,0,np.amin(mask_coor_vp_angles))
                surface_angles = np.append(surface_angles,np.amax(mask_coor_vp_angles))

                for i in range(len(surface_angles)-1):
                    good = np.logical_and(mask_coor_vp_angles>=surface_angles[i], mask_coor_vp_angles<=surface_angles[i+1])
                    if np.count_nonzero(good) > 0:
                  
                        slice = mask_coor[0][good], mask_coor[1][good]

                        # surface.probs[slice]= np.mean(surface.probs[slice])
                        # surface_img[slice] = np.mean(self.room.normals[slice], axis=0)
                
                #best_lines_1 = merge_lines(best_lines_1, search_width = self.diagonal / 400, search_length=1.2, angle_threshold=np.radians(5))
                surface.barriers.append(SurfaceBarrier(vps[vp_index_1], best_lines_1))

                draw_lines(surface_img, best_lines_1, color=0, thickness=3,lineType=cv2.LINE_AA)
                draw_lines(surface_img, best_lines_1, color=color2, thickness=2)
            
            if best_lines_2 is not None:
                color3 = colors[vp_index_2]
                vp_pt = vps[vp_index_2].model
                vp_pt = vp_pt[:2]/vp_pt[2]
                mask_coor_vp_x = mask_coor[1] - vp_pt[0]
                mask_coor_vp_y = mask_coor[0] - vp_pt[1]
                mask_coor_vp_angles = np.arctan2(mask_coor_vp_y, mask_coor_vp_x)
                surface_angles = [line.angle for line in best_lines_2]
                surface_angles = np.insert(surface_angles,0,surface_angles[0] - np.pi/4)
                surface_angles = np.append(surface_angles,surface_angles[-1] + np.pi/4)
                if len(surface_angles) == 0:
                    continue
                for i in range(len(surface_angles)-1):
                    good = np.logical_and(mask_coor_vp_angles>=surface_angles[i], mask_coor_vp_angles<=surface_angles[i+1])
                    if np.count_nonzero(good) > 0:
                        slice = mask_coor[0][good], mask_coor[1][good]

                        surface.probs[slice]= np.mean(surface.probs[slice])
                        # surface_img[slice] = np.mean(surface_img[slice], axis=0)
                #best_lines_2 = merge_lines(best_lines_2, search_width = self.diagonal / 400, search_length=1.2, angle_threshold=np.radians(5))
                surface.barriers.append(SurfaceBarrier(vps[vp_index_2], best_lines_2))

                draw_lines(surface_img, best_lines_2, color=0, thickness=4,lineType=cv2.LINE_AA)
                draw_lines(surface_img, best_lines_2, color=color3, thickness=2)
            # draw_lines(surface_img, lines, color=0, thickness=2,lineType=cv2.LINE_AA)
        
            log_image(self.data, "surface_img" + str(cluster_index), surface_img)
            # log_image(self.data, "surface_probs_vp_refinded" + str(cluster_index),255.*surface.probs)
            cluster_index+=1

    def find_best_vanishing_points(self, surfaceType, mask, mask_center):

        candidates = self.room.get_surfaces([surfaceType])
        
        if len(candidates) == 0:
            return None

        if len(candidates) > 1:

            mask_sample = sample_at_point(mask, point=mask_center)

            if cv2.countNonZero(mask_sample) > 0:
                sample = sample_at_point(self.room.normals, point=mask_center)
                normal = cv2.mean(sample, mask_sample)[:3]
                candidates.sort(key=lambda x: distance.sqeuclidean(mask_center, x.center) * distance.sqeuclidean(normal, x.normals_color))
            else:
                candidates.sort(key=lambda x: distance.sqeuclidean(mask_center, x.center))

        return candidates[0]
    
    # def get_contour_lines(self, image, surface, min_length):
    #     lines = []

    #     for poly in surface.polygons:
    #         num_pts = len(poly)

    #         for i in range(num_pts):
    #             point_a = poly[i][0]
    #             point_b = poly[(i+1) % num_pts][0]

    #             if line_on_image_edge(point_a, point_b, image.shape[1], image.shape[0]):
    #                 continue

    #             if distance.euclidean(point_a, point_b) > min_length:
    #                 lines.append(Line(point_b[0], point_b[1], point_a[0], point_a[1]))
    
    #     return lines

    def find_best_surface(self, surfaceType, mask, mask_center=None):

        candidates = self.room.get_surfaces([surfaceType])
        
        if len(candidates) == 0:
            return None

        if len(candidates) > 1:
            if mask_center is None:
                # mask_center = np.argmax(np.sum(mask, axis=1))
                mask_center = np.mean(np.nonzero(mask),0)

            mask_sample = sample_at_point(mask, point=mask_center)

            if cv2.countNonZero(mask_sample) > 0:
                sample = sample_at_point(self.room.normals, point=mask_center)
                normal = cv2.mean(sample, mask_sample)[:3]
                candidates.sort(key=lambda x: distance.sqeuclidean(mask_center, x.center) * distance.sqeuclidean(normal, x.normals_color))
            else:
                candidates.sort(key=lambda x: distance.sqeuclidean(mask_center, x.center))

        return candidates[0]

    def add_missing_surfaces(self, invalid_mask=None, min_area=1/2500):

        total_area = self.room.image.shape[0] * self.room.image.shape[1]
        area_threshold = int(total_area * min_area)

        #print("Size threshold: square greater than %d pixels on one side" % np.sqrt(area_threshold))

        if len(self.room.surfaces) > 0:
            total_mask = np.sum(np.dstack([s.mask for s in self.room.surfaces]), axis=-1)
        else:
            total_mask = None

        room_missing = self.room.image.copy() if im_logging_enabled(self.data) else None

        total_elevation = 3 #todo: get total elevation from highest and lowest objects. Floor or ceiling could be missing

        for surfaceType in SurfaceType:
            # if surfaceType == SurfaceType.Wall:
            #     continue

            color = random_color()

            #find missing
            remaining_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)
            remaining_mask[self.room.isolated_labels == surfaceType] = 1
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

                contour_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)
                cv2.drawContours(contour_mask, [contour], 0, (1,1,1), cv2.FILLED)

                matches = self.room.index_mask[contour_mask > 0]
                indexes, counts = np.unique(matches, return_counts=True)
                
                valid_clusters = indexes[counts > area_threshold]

                max_clusters = len(valid_clusters)
                new_surface = None
                reference_surface = None

                #add missing one by one
                for i in range(max_clusters):
                    index = valid_clusters[i]
                    if index >= len(self.room.surfaces):
                        continue

                    surface = self.room.surfaces[index]

                    if surface.surfaceType == surfaceType or surface.surfaceType.is_pair(surfaceType):

                        mask = contour_mask.copy()
                        # mask[self.room.index_mask != index] = 0

                        if cv2.countNonZero(mask) >= area_threshold:

                            reference_surface = surface
                            new_surface = reference_surface.clone()
                            new_surface.surfaceType = surfaceType
                            new_surface.mask = mask
                            break

                    elif not surface.bestLabel:
                        surface.destroyed = True
                        break

                new_normal = None
                new_offset = None

                if new_surface is None:

                    moments = cv2.moments(contour_mask)

                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])

                    reference_surface = self.find_best_surface(surfaceType, contour_mask, (cX, cY))
                    
                    if reference_surface is not None:
                        new_surface = reference_surface.clone()
                        new_surface.surfaceType = surfaceType
                        new_surface.mask = contour_mask
                        #print("Reference_surface", reference_surface.name, indexes, counts)
                    elif surfaceType in (SurfaceType.Floor, SurfaceType.OnFloor, SurfaceType.Ceiling):
                        opposing_type = SurfaceType.Floor if surfaceType == SurfaceType.Ceiling else SurfaceType.Ceiling
                        reference_surface = self.find_best_surface(opposing_type, contour_mask, (cX, cY))

                        if reference_surface is None:
                            #use onfloor->floor
                            reference_surface = self.find_best_surface(surfaceType.complement, contour_mask, (cX, cY))
                            if reference_surface is not None:
                                normal = reference_surface.normal
                                offset = reference_surface.offset
                        else:
                            normal = -reference_surface.normal
                            offset = total_elevation - reference_surface.offset

                        if reference_surface is not None:
                            print("Generate %s using %s as opposing surface" % (surfaceType.name, reference_surface.name))
                            
                            new_surface = reference_surface.clone()
                            new_surface.surfaceType = surfaceType
                            new_surface.mask = contour_mask
                            new_surface.normal = normal
                            new_surface.offset = offset


                    if reference_surface is None:
                        print(colored("No match for %s" % surfaceType.name, 'yellow'))       

                if new_surface is not None:
                    print(colored("Creating new %s (%s) using %s as reference" % (surfaceType.name, new_surface.name, reference_surface.name), 'green'))
                    self.room.add_surface(new_surface)
                    

        self.room.invalidate()
        
    def refine_surfaces(self, min_confidence=None, freedom=0.33, use_lines=True, debug_suffix=""):
        watershed_image = cv2.resize(self.data["hed"], (self.room.image.shape[1], self.room.image.shape[0]))
        diagonal = self.diagonal
        # print("diagonal", diagonal)
        
        def expand_into_type(surfaceType:SurfaceType):
            surfaces = self.room.get_surfaces([surfaceType])

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
                markers[dist_transform > freedom * dist_transform.max()] = index + 1

            markers[disputed_areas > 0] = 0

            watershed_mask = np.zeros(total_mask.shape, dtype=np.int32)
            watershed_mask[self.room.isolated_labels == surfaceType.index] = 1
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
                if use_lines:
                    mask[self.lines_mask > 0] = 0

                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
                mask = cv2.dilate(mask, kernel)
                mask = remove_small_holes(mask, area_threshold=surface.max_area/50).astype(np.uint8)

                if min_confidence is not None: 
                    mask[surface.probs < min_confidence] = 0
                
                if surface.surfaceType == SurfaceType.Wall:
                    n, mask_components, stats = cv2.connectedComponentsWithStats(mask.astype(np.uint8))[:3]
                    if n > 1:
                        max_area = np.amax(stats[1:, cv2.CC_STAT_AREA])
                
                        for i in range(1, n):

                            area = stats[i, cv2.CC_STAT_AREA]
            
                                # print(colored("%i: %s: %i: %i" % (max_area, surface.name, area, i), 'yellow'))
                            threshold = min(max_area * 0.1,100)

                            if area > threshold:
                                new_mask = np.uint8(mask_components==i)

                                new_surface = surface.clone()
                                new_surface.surfaceType = surface.surfaceType
                                new_surface.mask = new_mask

                                self.room.add_surface(new_surface)

                else: 

                    new_surface = surface.clone()
                    new_surface.surfaceType = surface.surfaceType
                    new_surface.mask = mask

                    self.room.add_surface(new_surface)


                surface.destroy()

        #expand all surfaces as far as they can go within their segmentation (watershed)
        #and resolve disputes between planes as they intersect by probability (confidence):

        for surfaceType in SurfaceType: 
            expand_into_type(surfaceType)

        self.room.refresh_surfaces()
        

    def remove_invalid_surfaces(self, min_area_threshold=1/1000, max_area_threshold=1/50, scale=1.2):

        total_area = self.room.image.shape[0] * self.room.image.shape[1]
        max_area = total_area * max_area_threshold
        min_area = total_area * min_area_threshold
        
        all_invalid_contours = []
        print("Remove invalid wall parts.")

        invalid_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)

        for surface in self.room.get_surfaces([SurfaceType.Wall]):
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
                # else:
                #     inner_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)
                #     cv2.drawContours(inner_mask, np.array(contour), 0, 1, cv2.FILLED)
                    
                #     innerMean, innerStd = cv2.meanStdDev(self.room.image, mask=inner_mask)

                #     outer_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)
                #     contour_expanded = scale_contour(contour, scale, moments=M)
                #     cv2.drawContours(outer_mask, np.array(contour_expanded), 0, 1, cv2.FILLED)
                #     #outer_mask[inner_mask] = 0
                #     outerMean, outerStd = cv2.meanStdDev(self.room.image, mask=outer_mask)

                #     meanDiff = np.max(np.abs(innerMean - outerMean))

                #     #really want the standard deviation here, but opencv is returning ZEROS:
                #     #threshold = meanDiff + innerStd * meanDiff

                #     print("surface %s(%d) %.2f" % (surface.name, i, meanDiff))

                #     if meanDiff < 5:
                #         #print("remove %s" % name, meanDiff)
                #         invalid_contours.append(contour)
                   
            all_invalid_contours.extend(invalid_contours)

            if len(invalid_contours) == len(surface.contours):
                surface.destroy() #totally invalid
                print(colored("Removing surface %s" % surface.name, "red"))
            elif len(invalid_contours) > 0:
                print(colored("Removing %d contours from surface %s" % (len(invalid_contours), surface.name), "yellow"))
                mask = surface.mask.copy()
                cv2.drawContours(mask, np.array(invalid_contours), -1, 0, cv2.FILLED)
                surface.mask = mask

                #todo: look inside contour for validity



        if len(all_invalid_contours) > 0:
            cv2.drawContours(invalid_mask, np.array(all_invalid_contours), -1, 1, cv2.FILLED)
            self.room.refresh_surfaces()

        return invalid_mask


    def assign_parents(self):

        child_surfaces = self.room.get_surfaces(surfaceTypes=[SurfaceType.OnWall, SurfaceType.OnFloor, SurfaceType.OnCeiling, SurfaceType.Other])

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


class PipelineSurfaceSolver(PipelineStep):
    @property
    def index(self) -> PipelineStepIndex:
        return PipelineStepIndex.SolveSurfaces

    @property
    def required_keys(self) -> list:
        return ["room"]

    @property
    def output_keys(self) -> list:
        return []

    def run(self, data):
        solver = SurfaceSolver(data)
        data["room"] = solver.solve()

