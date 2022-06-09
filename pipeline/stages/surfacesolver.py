from re import M
from matplotlib import lines
import numpy as np
from scipy import ndimage
import cv2
from termcolor import colored
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from scipy.spatial import distance

from pipeline.core import PipelineStep, PipelineStepIndex 
from pipeline.data.surface_type import SurfaceType
from pipeline.misc.utils import resize_array, random_color, overlay_mask, sample_at_point, scale_contour
from .planegeometry import Dimension
from pipeline.data.logging import log_image, log_segmentation_image, im_logging_enabled, log_markers, Timer
from pipeline.components.line import Line, draw_lines, line_on_image_edge
from pipeline.components.surface import Surface
from pipeline.data.ade20k import ADE20K
from pipeline.stages.vanishingpointfinder import get_votes

class SurfaceSolver():

    def __init__(self, data):
        super().__init__()
        self.data = data
        self.room = self.data["room"]

    def solve(self):
        
        #add all the applicable surfaces:
        timer = Timer("room")

        #prepare
        self.lines_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)
        draw_lines(self.lines_mask, self.data["lines"], color=255, thickness=2, lineType=cv2.LINE_AA)

        self.refine_surfaces()
  
        #preserve plane context information i.e. probs < min_confidence are ignored

        log_image(self.data, "room_refined", self.room.get_debug_image())
        timer.time_event("refine_surfaces")
 
        # invalid_mask = self.remove_invalid_surfaces()
        # invalid_mask[self.lines_mask > 0] = 255

        # if im_logging_enabled(self.data) and cv2.countNonZero(invalid_mask) > 50:
        #     debug = self.room.image.copy()
        #     debug[invalid_mask > 0] = [0,255,0]
        #     log_image(self.data, "room_removed", debug)
        
    
        # timer.time_event("remove_invalid_surfaces")

        # self.add_missing_surfaces(invalid_mask,1/2400)
        # log_image(self.data, "room_missing_added", self.room.get_debug_image())

        # timer.time_event("add_missing_surfaces")

        # self.merge_like_surfaces()

        # log_image(self.data, "room_merged", self.room.get_debug_image())

        # self.refine_surfaces(debug_suffix="_final")
        # self.refine_surfaces()

        # log_image(self.data, "room_expanded", self.room.get_debug_image())
      
        
        # timer.time_event("room.refine_surfaces")

        # self.finalize_masks(invalid_mask)
       
        # log_image(self.data, "room", self.room.get_debug_image())

        surfaces = []
        for surfaceType in SurfaceType: 
            surfaces.extend(self.data["room"].get_surfaces([surfaceType]))

        vertical_vp = self.room.vertical_vp
        vertical_lines = vertical_vp.inliers.copy()

        self.height, self.width = self.room.image.shape[:2]
        diagonal = np.hypot(self.width, self.height)

        surface_image = self.room.image.copy()
 
        vertical_labels = self.room.isolated_labels.copy()

        # vertical_labels[self.lines_mask > 0] = 0
        isolated_probs = self.data["isolated"]

        for surfaceType in [SurfaceType.Wall, SurfaceType.OnWall]:
            label = surfaceType.value
            refined_type = np.sum([surface.mask for surface in surfaces if surface.surfaceType == surfaceType],0)
            vertical_labels[np.logical_and(vertical_labels == label, refined_type==0)] = -1
            vertical_labels[refined_type>0]= label

            mask = vertical_labels == label

            type_surface = Surface(self.data)
            type_surface.set_mask(np.uint8(mask))
            type_lines = type_surface.lines

            type_lines = filter(lambda x: x in vertical_lines, type_lines)

            type_lines =  list(map(lambda line: line.extended(1, vanishing_point=vertical_vp.model[:2]), type_lines))
            if len(type_lines) == 0:
                continue

            mask_coor = np.nonzero(mask)
                   
            mask_coor_vp_x = mask_coor[1] - vertical_vp.model[:2][0]
            mask_coor_vp_y = mask_coor[0] - vertical_vp.model[:2][1]
            # innefficient
            mask_coor_vp_angles = np.arctan2(mask_coor_vp_y, mask_coor_vp_x) % np.pi

            type_lines.sort(key=lambda x:x.angle%np.pi)
            surface_angles = [line.angle%np.pi for line in type_lines]
            surface_angles = np.insert(surface_angles,0,np.amin(mask_coor_vp_angles))
            surface_angles = np.append(surface_angles,np.amax(mask_coor_vp_angles))
            pixel_angle_width = abs((surface_angles[-1]- surface_angles[0])/(np.amax(mask_coor_vp_x)-np.amin(mask_coor_vp_x)))
            print("p", np.degrees(pixel_angle_width))
            start_angle = surface_angles[0]
            current_normal = None

            for i in range(len(surface_angles)-1):
                # i = len(surface_angles)-2 - i

                good = np.logical_and(mask_coor_vp_angles >=surface_angles[i], mask_coor_vp_angles < surface_angles[i+1]) 
                if np.count_nonzero(good) > 0:

                    slice = mask_coor[0][good], mask_coor[1][good]

                    ang_diff = abs(surface_angles[i+1] - surface_angles[i])
                    if ang_diff < 3.*pixel_angle_width:
                        vertical_labels[slice] = np.amax(vertical_labels)
                    else:
                        slice_normal = np.mean(self.room.normals[slice], axis=0)
                        slice_normal = (slice_normal-127.5)/127.5
                        slice_normal /= np.linalg.norm(slice_normal)

                        if current_normal is None:
                            current_normal = slice_normal
                        else:
                            # print(current_normal, slice_normal,np.linalg.norm(current_normal))
                            cos_normal = np.dot(current_normal, slice_normal)
                            angle = np.arccos(cos_normal)
                            if abs(angle) > np.radians(1):
                                vertical_labels[slice] = np.amax(vertical_labels) + 1
                            else:
                                vertical_labels[slice] = np.amax(vertical_labels)

                            current_normal = slice_normal


        log_segmentation_image(self.data, "isolated_labels_normals", vertical_labels, self.room.normals, get_image=False, labelset=SurfaceType,opacity=1, avg=True)

        log_segmentation_image(self.data, "isolated_labels_image", vertical_labels, self.room.image, get_image=False, labelset=SurfaceType,opacity=.9, avg=False)

           
        timer.time_event("setup")
        

     
        # for label in np.unique(isolated_labels):
        #     if label != 2 and label != 3: continue

        #     mask = isolated_labels == label
        #     mask_coor = np.nonzero(mask)
                   
        #     mask_coor_vp_x = mask_coor[1] - vertical_vp_pt[0]
        #     mask_coor_vp_y = mask_coor[0] - vertical_vp_pt[1]
        #     mask_coor_vp_angles = np.arctan2(mask_coor_vp_y, mask_coor_vp_x) % np.pi

        #     surfaces_current = [surface for surface in surfaces if surface.surfaceType.value == label]

        #     for j in range(len(surfaces_current)):
        #         surface = surfaces_current[j]
        #         surface_angles = [line.angle%np.pi for line in vertical_vp.inliers if line in surface.lines]
        #         surface_angles = np.insert(surface_angles,0,np.amin(mask_coor_vp_angles))
        #         surface_angles = np.append(surface_angles,np.amax(mask_coor_vp_angles))
        #         for i in range(len(surface_angles)-1):
        #             good = np.logical_and(mask_coor_vp_angles>surface_angles[i], mask_coor_vp_angles<=surface_angles[i+1])
        #             if np.count_nonzero(good) > 0:

        #                 slice = mask_coor[0][good], mask_coor[1][good]
        #                 slice_prob = np.mean(surface.probs[slice])
        #                 surface.probs[slice] = slice_prob
                     
        #                 surface_image[slice] = random_color()
                  
        # log_image(self.data, "surfacessssss", surface_image)

        # self.refine_surfaces(min_confidence=.01)

   

    

        surfaces = []
        for surfaceType in SurfaceType: 
            surfaces.extend(self.data["room"].get_surfaces([surfaceType]))

        cluster_index = 0
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 0, 255)]
        for c in range(100):
            colors.append(random_color())
   
        vps =  self.room.horizontal_vps
        vps = np.insert(vps, 0, self.room.vertical_vp)
        surface_img = self.room.image.copy()

        for surface in surfaces:
            if surface.surfaceType == SurfaceType.Other:
                continue

            lines = surface.lines
            if len(lines) == 0:
                continue    

            max_score_1 = 0
            max_score_2 = 0
            best_lines_1 = None
            best_lines_2 = None
            vp_index_1 = None
            vp_index_2 = None

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

                print("Surface %s has %f votes (max scores %f, %f) from vps %i:" % (surface.name, score,max_score_1, max_score_2, i))
            
            if vp_index_1 is not None and vp_index_2 is not None:
                print("vp_index_1: %i, vp_index_2: %i" % (vp_index_1, vp_index_2))

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
                        surface_img[slice] = np.mean(self.room.normals[slice], axis=0)
      

                draw_lines(surface_img, best_lines_1, color=0, thickness=4,lineType=cv2.LINE_AA)
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
                        surface_img[slice] = np.mean(surface_img[slice], axis=0)


                draw_lines(surface_img, best_lines_2, color=0, thickness=4,lineType=cv2.LINE_AA)
                draw_lines(surface_img, best_lines_2, color=color3, thickness=2)

        
            log_image(self.data, "surface_img" + str(cluster_index), surface_img)
            log_image(self.data, "surface_probs_vp_refinded" + str(cluster_index),255.*surface.probs)
            cluster_index+=1




        
        # for j in range(len(surfaces)):
        #     color = random_color()
        #     img_surface = self.room.image.copy()
        #     surface = surfaces[j]
        #     if surface.surfaceType != SurfaceType.Wall and surface.surfaceType != SurfaceType.OnWall:
        #         continue
        #     # if surface.surfaceType != SurfaceType.Wall:
        #     #     continue

        #     reference_surface = surface
        #     mask = np.uint8(isolated_labels==surface.surfaceType.value)
        #     draw_lines(surface.probs, surface.lines, color=0, thickness=1, lineType=cv2.LINE_AA)
        #     # draw_lines(mask, surface.lines, color=0, thickness=2, lineType=cv2.LINE_AA)
        #     mask_coor = np.nonzero(mask)

        #     mask_coor_vp_x = mask_coor[1] - vertical_vp_pt[0]
        #     mask_coor_vp_y = mask_coor[0] - vertical_vp_pt[1]
        #     mask_coor_vp_angles = np.arctan2(mask_coor_vp_y, mask_coor_vp_x) % np.pi

        #     surface_lines = [line for line in vertical_vp.inliers if line in surface.lines]
            
        #     xys = np.array([line.data for line in surface.lines if line not in vertical_vp.inliers])

        #     xys = xys.reshape(-1,2)
        #     if len(xys)>0:
        #         pt = xys[np.argmax(xys[:,0])]
        #         cv2.line(img_surface, (int(pt[0]), int(pt[1])), (int(vertical_vp_pt[0]), int(vertical_vp_pt[1])), color=255, thickness=4)

           

        #     surface_angles = [line.angle%np.pi for line in surface_lines if line in surface.lines]
        #     if len(surface_angles) == 0:
        #         continue
        #     # surface_angles = np.insert(surface_angles,0,surface_angles[0] - np.pi/4)
        #     # surface_angles = np.append(surface_angles,surface_angles[-1] + np.pi/4)
        #     min_angle = np.amin(mask_coor_vp_angles)
        #     max_angle = np.amax(mask_coor_vp_angles)
        #     surface_angles = np.insert(surface_angles,0,min_angle)
        #     surface_angles = np.append(surface_angles,max_angle)
        #     surface_angles = np.array(surface_angles)
        #     small = np.nonzero(surface_angles<min_angle)[0]
        #     big = np.nonzero(surface_angles>max_angle)[0]

        #     first = 0
        #     last = len(surface_angles)-1
        #     if len(small) > 0:
        #         first = max(small[-1],0)
        #     if len(big) > 0:
        #         last = min(big[-1]+1,len(surface_angles)-1)
        #     print(first, last,"small", small, "big", big)
        #     # surface_angles = surface_angles[first:last//2]
        #     # surface_angles = surface_angles[surface_angles>min_angle]
        #     # surface_angles = surface_angles[surface_angles<max_angle]
        #     # surface.mask[mask_coor[0][surface_angles>min_angle], mask_coor[1][surface_angles>min_angle]] = 1 
        #     # print(surface_angles)
        #     # surface.mask[:,:] = 0

        #     # surface_angles = np.insert(surface_angles,0,np.amin(mask_coor_vp_angles))
        #     # surface_angles = np.append(surface_angles,np.amax(mask_coor_vp_angles))

        #     for i in range(len(surface_angles)-1):
        #         slice_color = random_color()
        #         good = np.logical_and(mask_coor_vp_angles>surface_angles[i], mask_coor_vp_angles<surface_angles[i+1])
        #         if np.count_nonzero(good) > 0:
        #             # print(surface_angles[i], surface_angles[i+1], np.count_nonzero(good))

        #             slice = mask_coor[0][good], mask_coor[1][good]
        #             slice_prob = np.mean(surface.probs[slice])
        #             surface.probs[slice] = slice_prob
        #             slice_mask_prob = np.mean(surface.mask[slice])
        #             # print(slice_mask_prob, slice_prob)
        #             # surface.mask[slice] = np.mean(surface.mask[slice])

        #             if slice_prob < 0.05:
        #                 surface.probs[slice] = 0
        #                 # surface.mask[slice] = 0
                 
        #                 # surface.mask[slice] = slice_prob
        #                 # mask = np.zeros_like(surface.mask)
        #             # if slice_prob > 0.01:
        #             #     img_surface[slice] = random_color()
        #             #     surface.mask[slice] = slice_mask_prob
        #             else:
                    
        #                 img_surface[slice] = slice_color
        #                 surface.mask[slice] = 1
                    
                        
        #                 # mask[slice] = 1
        #                 # new_surface = reference_surface.clone()
        #                 # new_surface.surfaceType = SurfaceType.Wall
        #                 # new_surface.set_mask(mask)

        #                 # self.room.add_surface(new_surface)
        #     draw_lines(img_surface, [line.data for line in surface.lines if line not in surface_lines], color=0, thickness=3, lineType=cv2.LINE_AA)

        #     log_image(self.data, str(j) + "_img_surface", img_surface)
        #     log_image(self.data, str(j) + "_surface_probs", 255.*surface.probs)
        #     log_image(self.data, str(j) + "_surface_mask", 255.*surface.mask)

   


        # surfaces = []

        # for surfaceType in SurfaceType: 
        #     surfaces.extend(self.data["room"].get_surfaces([surfaceType]))

        # surface_img= self.room.image.copy()
        # for surface in surfaces:
        #     # lines.extend(self.get_contour_lines(image, surface, diagonal/80))
            
        #     if surface.cloned_from>-1 and surface.surfaceType==SurfaceType.Wall:
        #         mask_coor = np.nonzero(surface.mask)

        #         mask_coor_vp_x = mask_coor[1] - vertical_vp_pt[0]
        #         mask_coor_vp_y = mask_coor[0] - vertical_vp_pt[1]
        #         mask_coor_vp_angles = np.arctan2(mask_coor_vp_y, mask_coor_vp_x) % np.pi
        #         if len(mask_coor_vp_angles)==0:
        #             continue

        #         surface_lines = [line for line in vertical_vp.inliers if line in surface.lines]

        #         surface_angles = [line.angle%np.pi for line in surface_lines if line in surface.lines]

        #         surface_angles = np.insert(surface_angles,0,np.amin(mask_coor_vp_angles))
        #         surface_angles = np.append(surface_angles,np.amax(mask_coor_vp_angles))
        #         if len(surface_angles) == 0:
        #             continue
                
    
        #         for i in range(len(surface_angles)-1):
        #             good = np.logical_and(mask_coor_vp_angles>surface_angles[i], mask_coor_vp_angles<surface_angles[i+1])
        #             if np.count_nonzero(good) > 0:
        #                 color = random_color()
        #                 # print(surface_angles[i], surface_angles[i+1], np.count_nonzero(good))

        #                 slice = mask_coor[0][good], mask_coor[1][good]
        #                 # slice_prob = np.mean(surface.probs[slice])
        #                 # surface.probs[slice] = np.mean(surface.probs[slice])
        #                 # surface_img[slice] = np.mean(self.room.normals[slice], axis=0)
                    
        #                 # s /= np.amax(surface_angles)-np.amin(surface_angles)
                        
        #                 surface_img[slice] = color

        
        # log_image(self.data, "cloned", surface_img)



    

   




        
   
       


        timer.time_event("merge_like_surfaces")

        self.assign_parents()

        timer.time_event("assign_parents")

        timer.log_all_events()

        return self.room
    
    def get_contour_lines(self, image, surface, min_length):
        lines = []

        for poly in surface.polygons:
            num_pts = len(poly)

            for i in range(num_pts):
                point_a = poly[i][0]
                point_b = poly[(i+1) % num_pts][0]

                if line_on_image_edge(point_a, point_b, image.shape[1], image.shape[0]):
                    continue

                if distance.euclidean(point_a, point_b) > min_length:
                    lines.append(Line(point_b[0], point_b[1], point_a[0], point_a[1]))
    
        return lines

    def find_best_surface(self, surfaceType, mask, mask_center):

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

    def add_missing_surfaces(self, invalid_mask=None, min_area=1/1200):

        total_area = self.room.image.shape[0] * self.room.image.shape[1]
        area_threshold = int(total_area * min_area)

        print("Size threshold: square greater than %d pixels on one side" % np.sqrt(area_threshold))

        if len(self.room.surfaces) > 0:
            total_mask = np.sum(np.dstack([s.mask for s in self.room.surfaces]), axis=-1)
        else:
            total_mask = None

        room_missing = self.room.image.copy() if im_logging_enabled(self.data) else None

        total_elevation = 3 #todo: get total elevation from highest and lowest objects. Floor or ceiling could be missing

        for surfaceType in SurfaceType:
            
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
                            new_surface.set_mask(mask)
                            break

                    elif not surface.bestLabel:
                        surface.destroyed = True
                        break

                new_normal = None
                new_offset = None

                if new_surface is None and surfaceType.is_major:

                    moments = cv2.moments(contour_mask)

                    cX = int(moments["m10"] / moments["m00"])
                    cY = int(moments["m01"] / moments["m00"])

                    reference_surface = self.find_best_surface(surfaceType, contour_mask, (cX, cY))
                    
                    if reference_surface is not None:
                        new_surface = reference_surface.clone()
                        new_surface.surfaceType = surfaceType
                        new_surface.set_mask(contour_mask)
                        #print("Reference_surface", reference_surface.name, indexes, counts)
                    elif surfaceType == SurfaceType.Floor or surfaceType == SurfaceType.Ceiling:
                        complementary_type = SurfaceType.Floor if surfaceType == SurfaceType.Ceiling else SurfaceType.Ceiling
                        reference_surface = self.find_best_surface(complementary_type, contour_mask, (cX, cY))

                        if reference_surface is not None:
                            print("Generate %s using %s as opposing surface" % (surfaceType.name, reference_surface.name))
                            
                            new_surface = reference_surface.clone()
                            new_surface.surfaceType = surfaceType
                            new_surface.set_mask(contour_mask)
                            new_surface.normal = -reference_surface.normal
                            new_surface.offset = total_elevation - reference_surface.offset


                    if reference_surface is None:
                        print(colored("No match for %s" % surfaceType.name, 'yellow'))       

                if new_surface is not None:
                    print(colored("Creating new %s (%s) using %s as reference" % (surfaceType.name, new_surface.name, reference_surface.name), 'green'))
                    self.room.add_surface(new_surface)
                    

        self.room.invalidate()

    def merge_like_surfaces(self, angle_threshold=np.radians(5), angle_threshold_force=np.radians(5)):

        for surfaceType in SurfaceType:

            # if surfaceType == SurfaceType.Wall or surfaceType == SurfaceType.OnWall:
            #     continue

            color = random_color()
            surfaces = self.room.get_surfaces([surfaceType])

            for i in range(len(surfaces)):
                
                if surfaces[i].destroyed: continue
                # print(surfaces[i].bestLabel)

                distance_i = abs(surfaces[i].offset) #todo: calculate this?

                for j in range(i+1, len(surfaces)):

                    if surfaces[j].destroyed: continue

                    if surfaces[i].bestLabel != surfaces[j].bestLabel: 
                        # print(colored("Surfaces %s and %s are not in the same cluster" % (surfaces[i].name, surfaces[j].name), 'yellow'))
                        continue

                    dot_product = np.dot(surfaces[i].normal, surfaces[j].normal)
                    angle = np.arccos(dot_product)

                    #do some planar geometry comparisons, maybe color/texture
                    distance_j = abs(surfaces[j].offset)

                    distance_between = abs(distance_i - distance_j)
                    distance_mean = 0.5 * (distance_i + distance_j)
                    distance_error = 0.35 * distance_mean #accuracy degrades by range (maybe use error here, error square?)

                    #todo: check for intersection. In elevator image, wall sitting out front is being incorrectly merged. if it's fairly parallel, don't
                    if surfaceType == SurfaceType.Floor or surfaceType == SurfaceType.Ceiling or (angle < angle_threshold and distance_between < distance_error):
                        if surfaceType != SurfaceType.Other or surfaces[i].bestLabel == surfaces[j].bestLabel:
                            offset_diff = abs(surfaces[i].offset - surfaces[j].offset) / max(abs(surfaces[i].offset), abs(surfaces[j].offset))
                            
                            #print("angle", angle, "offset", offset_diff)

                            if offset_diff < 0.05 or angle < angle_threshold_force: 
                                surfaces[i].merge(surfaces[j])
                                surfaces[i]._alteration = "%.2fm %.2fd" % (distance_between, angle_threshold)

        
    def refine_surfaces(self, min_confidence=None, freedom=0.33, use_lines=True, debug_suffix=""):
        watershed_image = cv2.resize(self.data["hed"], (self.room.image.shape[1], self.room.image.shape[0]))

        final_masks = {}
        
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
                mask[self.lines_mask > 0] = 0
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(2,2))
                mask = cv2.dilate(mask, kernel)

                if min_confidence is not None: 
                    mask[surface.probs < min_confidence] = 0
  
                surface.set_mask(mask)

        #expand all surfaces as far as they can go within their segmentation (watershed)
        #and resolve disputes between planes as they intersect by probability (confidence):

        for surfaceType in SurfaceType: 
            expand_into_type(surfaceType)

        

    def remove_invalid_surfaces(self, min_area_threshold=1/1000, max_area_threshold=1/50, scale=1.2):

        total_area = self.room.image.shape[0] * self.room.image.shape[1]
        max_area = total_area * max_area_threshold
        min_area = total_area * min_area_threshold
        
        all_invalid_contours = []
        print("Remove invalid wall parts.")

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
                else:
                    inner_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)
                    cv2.drawContours(inner_mask, np.array(contour), 0, 1, cv2.FILLED)
                    
                    innerMean, innerStd = cv2.meanStdDev(self.room.image, mask=inner_mask)

                    outer_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)
                    contour_expanded = scale_contour(contour, scale, moments=M)
                    cv2.drawContours(outer_mask, np.array(contour_expanded), 0, 1, cv2.FILLED)
                    #outer_mask[inner_mask] = 0
                    outerMean, outerStd = cv2.meanStdDev(self.room.image, mask=outer_mask)

                    meanDiff = np.max(np.abs(innerMean - outerMean))

                    #really want the standard deviation here, but opencv is returning ZEROS:
                    #threshold = meanDiff + innerStd * meanDiff

                    print("surface %s(%d) %.2f" % (surface.name, i, meanDiff))

                    if meanDiff < 10:
                        #print("remove %s" % name, meanDiff)
                        invalid_contours.append(contour)
                   
            all_invalid_contours.extend(invalid_contours)

            if len(invalid_contours) == len(surface.contours):
                surface.destroy() #totally invalid
                print(colored("Removing surface %s" % surface.name, "red"))
            elif len(invalid_contours) > 0:
                print(colored("Removing %d contours from surface %s" % (len(invalid_contours), surface.name), "yellow"))
                mask = surface.mask.copy()
                cv2.drawContours(mask, np.array(invalid_contours), -1, 0, cv2.FILLED)
                surface.set_mask(mask)

                #todo: look inside contour for validity

        invalid_mask = np.zeros(self.room.image.shape[:2], dtype=np.uint8)

        if len(all_invalid_contours) > 0:
            cv2.drawContours(invalid_mask, np.array(all_invalid_contours), -1, 1, cv2.FILLED)
            self.room.refresh_surfaces()

        return invalid_mask

    def finalize_masks(self, invalid_mask=None):

        total_mask = []
        markers = np.zeros((self.room.image.shape[0], self.room.image.shape[1]), dtype=np.int32)
        if invalid_mask is not None:
            markers[invalid_mask > 0] = -1

        num_surfaces = len(self.room.surfaces)
        for index in range(num_surfaces):
            surface = self.room.surfaces[index]
            markers[surface.mask > 0] = index + 1

        markers = cv2.watershed(self.room.image, markers)
        markers[markers<0] = 0

        for index in range(num_surfaces):
            surface = self.room.surfaces[index]
            mask = np.zeros_like(surface.mask)
            mask[markers == (index + 1)] = 1
            # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

            surface.set_mask(mask)

        log_markers(self.data, "room_final_markers", markers)

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

