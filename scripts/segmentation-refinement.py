import math
import os
import json
import cv2
import numpy as np
from pathlib import Path

import click
import time

import io
from Line import Line
from frei_chen import frei_chen
import random

SAVE_DEBUG_IMAGES = True 

class WatershedLabel:
    BOUNDARY_LABEL = 255
    UNKNOWN_LABEL = 0
    OUTLIER_LABEL = 1

def translate_markers(debug_markers, markers, num_labels, hues):

    mask = np.zeros(markers.shape, dtype=np.uint8)
    mask[markers == WatershedLabel.OUTLIER_LABEL] = 255
    debug_markers = overlay_mask(debug_markers, mask, 0, saturation=0)

    for index in range(num_labels):
        label = WatershedLabel.OUTLIER_LABEL + 1 + index
        mask = np.zeros(markers.shape, dtype=np.uint8)
        mask[markers == label] = 255
        debug_markers = overlay_mask(debug_markers, mask, hues[index])

    debug_markers[markers == WatershedLabel.BOUNDARY_LABEL] = (0,0,255)

    return debug_markers

def rough_dilate_erode(is_dilate, mask, size=5, iterations=1, scale=0.5, maintain_size=True, interpolation=cv2.INTER_NEAREST):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(size,size))
    shape = mask.shape
    mask = cv2.resize(mask, (int(shape[1] * scale), int(shape[0] * scale)), interpolation)
    mask = cv2.dilate(mask, kernel, iterations=iterations) if is_dilate else cv2.erode(mask, kernel, iterations=iterations)
    if maintain_size:
        mask = cv2.resize(mask, (shape[1], shape[0]), interpolation)
    return mask

def convertColor(hsv, conversion):
    return tuple(int(i) for i in cv2.cvtColor(np.uint8([[hsv]]), conversion).flatten())

def overlay_mask(img, mask, hue=None, saturation=255, darkest_value=80):
    
    overlay = cv2.resize(mask, (img.shape[1], img.shape[0]))
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) #range 0-180

    grey = img_hsv[:, :, 2].copy()
    grey[grey<darkest_value] = darkest_value

    if hue is None:
        hue = random.randint(0,360)

    img_hsv[:, :, 0][overlay>0] = int(hue) / 2.0
    img_hsv[:, :, 1][overlay>0] = saturation
    img_hsv[:, :, 2][overlay>0] = grey[overlay>0] 

    out = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

    return out

def compute_edgelets(lines, class_labels=None):
    """Create edgelets as in the paper.
    Parameters
    ----------
    image: ndarray
        Image for which edgelets are to be computed.
    sigma: float
        Smoothing to be used for canny edge detection.
    Returns
    -------
    locations: ndarray of shape (n_edgelets, 2)
        Locations of each of the edgelets.
    directions: ndarray of shape (n_edgelets, 2)
        Direction of the edge (tangent) at each of the edgelet.
    strengths: ndarray of shape (n_edgelets,)
        Length of the line segments detected for the edgelet.
    """

    locations = []
    directions = []
    strengths = []
    classes = []

    for l in lines:
        l = l[0]
        p0, p1 = np.array([l[0], l[1]]), np.array([l[2], l[3]])
        c = (p0 + p1) / 2
        if classes:
            classes.append(class_labels[int(c[1]), int(c[0])])

        locations.append(c)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)
    classes = np.array(classes)

    directions = np.array(directions) / \
                 np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths, classes)

def edgelet_lines(edgelets):
    """Compute lines in homogenous system for edglets.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    Returns
    -------
    lines: ndarray of shape (n_edgelets, 3)
        Lines at each of edgelet locations in homogenous system.
    """
    locations, directions = edgelets[:2]
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines

def compute_votes(edgelets, model, threshold_inlier=math.radians(5)):
    """Compute votes for each of the edgelet against a given vanishing point.
    Votes for edgelets which lie inside threshold are same as their strengths,
    otherwise zero.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    model: ndarray of shape (3,)
        Vanishing point model in homogenous cordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        edgelet direction and line connecting the  Vanishing point model and
        edgelet location is used to threshold.
    Returns
    -------
    votes: ndarry of shape (n_edgelets,)
        Votes towards vanishing point model for each of the edgelet.
    """
    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets[:3]

    est_directions = locations - vp

    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
               np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = np.abs(dot_prod / abs_prod)

    theta_thresh = np.cos(threshold_inlier)

    return (cosine_theta > theta_thresh) * strengths

def ransac_vanishing_point(edgelets, num_ransac_iter=2000, threshold_inlier=math.radians(5), max_time=1.0, find_vert=True,
                           seeds=None):
    """Estimate vanishing point using Ransac.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.
    Returns
    -------
    best_model: ndarry of shape (3,)
        Best model for vanishing point estimated.
    Reference
    ---------
    Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
    """
    locations, directions, strengths = edgelets[:3]
    lines = edgelet_lines(edgelets)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = []
    best_votes = 0
    t = time.time()

    for ransac_iter in range(num_ransac_iter):
        if time.time() - t > max_time:
            return best_model

        ind1 = np.random.choice(first_index_space)

        # if classes[ind1]!=0 and classes[ind1]!=3: continue

        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model ** 2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue


        if find_vert:
            if current_model[1] / current_model[2] < 1000: continue

            dt1 = abs(np.dot(directions[ind1], [0, 1]))
            dt2 = abs(np.dot(directions[ind2], [0, 1]))

            if dt1 < .95 or dt2 < .95:
                continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier).sum()

        if np.any(seeds != None):
            cm = current_model[:2] / current_model[2]
            dot = 1 - abs(np.dot(cm / np.linalg.norm(cm), seeds[:2] / np.linalg.norm(seeds[:2])))

            current_votes = current_votes * dot > .9

        if np.any(current_votes > best_votes):
            # print("Current best model has {} votes at iteration {},{}".format(
            #     best_votes, ransac_iter, np.round(time() - t, 2)))

            best_model = current_model / current_model[2]
            best_votes = current_votes

    return best_model

def find_surfaces(img, surfaces, output_path):

    height, width = img.shape[:2]
    diagonal = np.hypot(width, height)
    print("image w,h", width, height)

    bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gabor_scale = 1500.0 / diagonal
    bw_res = cv2.resize(bw, (int(width * gabor_scale), int(height * gabor_scale)), cv2.INTER_CUBIC) if gabor_scale < 1.0 else bw

    def gabor(theta, lambd, gamma = 0.0, psi = 0.0):
        ksize = lambd
        sigma = ksize * lambd
        result = cv2.filter2D(bw_res, cv2.CV_8UC1, cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F))
        return result

    v_gabor = gabor(0, 7)
    h_gabor = gabor(np.pi/2.0, 9)

    contours_src = cv2.addWeighted(v_gabor, 1.0, h_gabor, 1.0, 0)
    contours_src = cv2.resize(contours_src, (width, height), interpolation = cv2.INTER_CUBIC)
    
    edges = cv2.addWeighted(v_gabor, 3.0, h_gabor, 3.0, -20)
    edges = cv2.bilateralFilter(edges, 5, 5, 5)
    edges = cv2.resize(edges, (width, height), interpolation = cv2.INTER_CUBIC)

    clean_edges = frei_chen(bw)

    line_data = []

    if SAVE_DEBUG_IMAGES:
        lines_a = img.copy()
        lines_b = img.copy()
        lines_c = img.copy()
        lines_d = img.copy()
        intersections = []
    else:
        lines_a = lines_b = lines_c = lines_d = None

    def is_image_edge(point_a, point_b, shape, dist=10):
        max_0 = shape[0] - 1
        max_1 = shape[1] - 1
        return (abs(point_a[0]) <= dist and abs(point_b[0]) <= dist) \
            or (abs(point_a[0] - max_0) <= dist and abs(point_b[0] - max_0) <= dist) \
            or (abs(point_a[1]) <= dist and abs(point_b[1]) <= dist) \
            or (abs(point_a[1] - max_1) <= dist and abs(point_b[1] - max_1) <= dist)

    def add_contour_lines(contours, min_confidence):
        epsilon = diagonal / 200.0
        min_length = diagonal / 40.0
        contour_group = 0
        for contour in contours:

            poly = cv2.approxPolyDP(contour, epsilon, False)
            contour_index = 0
            arcLen = cv2.arcLength(poly, False)

            for i in range(0, len(poly)-1):
                point_a = poly[i][0]
                point_b = poly[i+1][0]

                #remove contours on image edge and break them up into seperate contours (contour_group)
                if is_image_edge(point_a, point_b, (width, height), epsilon+1.0):
                    contour_index = 0
                    contour_group += 1
                elif arcLen > min_length:
                    new_line = Line(point_a[0], point_a[1], point_b[0], point_b[1], contour_group, contour_index)
                    line_data.append(new_line)
                    contour_index += 1

        contour_group += 1

    filtered_contours = []
    contours_src = cv2.adaptiveThreshold(contours_src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, int(diagonal / 50) * 2 + 1, -30)
    contours_dilated = rough_dilate_erode(True, contours_src, 3, scale=400/diagonal, interpolation=cv2.INTER_AREA)
    Line.prepare(img, contours_dilated, lines_c)

    #find all lines in the edge image
    fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 30.0), 1.41, 200, 240, 3, False)
    lines = fld.detect(edges)
    
    if lines is not None:
        for line in lines: 
            new_line = Line(line[0][0], line[0][1], line[0][2], line[0][3])
            if new_line.get_confidence() > 0.2: line_data.append(new_line)

    aperture = 5 if diagonal > 1200 else 3
    fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 50.0), 1.41, 200, 220, aperture, False)
    lines = fld.detect(bw - (clean_edges * 5.0).astype("uint8"))    
    
    if lines is not None:
        for line in lines: 
            new_line = Line(line[0][0], line[0][1], line[0][2], line[0][3])
            if new_line.get_confidence() > 0.15: line_data.append(new_line)
    
    if lines_a is not None: Line.draw_all(line_data, lines_a)
    
    line_data = Line.merge(line_data, diagonal / 150.0, search_length=1.0, angle_threshold=math.radians(3.0), max_color_std=5.0, color=(255,150,50))

    if lines_b is not None: Line.draw_all(line_data, lines_b)

    line_data = Line.merge(line_data, diagonal / 120.0, search_length=1.05, angle_threshold=math.radians(2.0), color=(255,0,0))

    if lines_c is not None: Line.draw_all(line_data, lines_c)

    line_data, intersections = Line.find_corners(line_data, search_length=1.5, angle_threshold=math.radians(10.0), parallel_threshold=math.radians(4), \
            max_color_std=3.0, confidence_diff=0.4)

    line_data = Line.merge(line_data, diagonal / 200.0, search_length=1.07, angle_threshold=math.radians(5.0), color=(255,0,0))

    if lines_d is not None: Line.draw_all(line_data, lines_d)


    #vanishing points:
    edgelets = Line.compute_edgelets(line_data)

    vp_thresh = math.radians(3)

    vp0 = ransac_vanishing_point(edgelets, 3000, threshold_inlier=vp_thresh, max_time=.5, find_vert=True)
    
    if len(vp0) > 0:
        vertical_line_inliers = compute_votes(edgelets, vp0, threshold_inlier=vp_thresh) > 0
        locations, directions, strengths, classes = edgelets
        edgelets = (locations[vertical_line_inliers], directions[vertical_line_inliers], strengths[vertical_line_inliers])
        locations, directions, strengths = edgelets
        vp_directions = locations - vp0[:2]

        angles = np.arctan2(vp_directions[:, 1], vp_directions[:, 0])

        s = np.argsort(angles)
        angles = angles[s]
        locations = locations[s]
        directions = directions[s]
        strengths = strengths[s]

        for i in range(-1, len(locations)):

            xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2.,
                   locations[i, 0] + directions[i, 0] * strengths[i] / 2.]
            yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2.,
                   locations[i, 1] + directions[i, 1] * strengths[i] / 2.]

            cv2.line(lines_d, (int(xax[0]), int(yax[0])), (int(xax[1]), int(yax[1])), color=(255,0,255), thickness=3)

    #cv2.imwrite(os.path.join(output_path, "vertical_edges.png"), 255. * vertical_edges)
    if lines_d is not None:

        for intersection in intersections: cv2.circle(lines_c, (int(intersection[0]), int(intersection[1])), max(int(diagonal/100), 3), (255,0,0), 2)
        
        cont_img = np.vstack((np.hstack((lines_a, lines_b)), np.hstack((lines_c, lines_d))))
        cv2.imwrite(os.path.join(output_path, "cont.jpg"), cont_img)

    return #line finding finished

    #refine mask:
    def get_dilation_amount(surface, amount, min_value=3, max_value=21):
        return max(min(int(amount / 2 + amount * surface["distance_factor"]), max_value), min_value)

    markers = np.zeros((height, width), dtype=np.int32)

    if SAVE_DEBUG_IMAGES:
        debug_before = img.copy()
        debug_analysis = img.copy()

    hues = random.sample(range(0, 350, 10), 1 + len(surfaces))
    scale = 500.0 / diagonal

    outliers_mask = np.ones(bw.shape, dtype=np.uint8) * 255

    for index in range(len(surfaces)):
        surface = surfaces[index]
        surface['hue'] = hues[index]
        surface['label'] = WatershedLabel.OUTLIER_LABEL + index + 1
        surface['mask'] = cv2.resize(surface['mask'], (width, height))

        #if surface['type'] in major_surface_types:
        if SAVE_DEBUG_IMAGES:
            debug_before = overlay_mask(debug_before, surface['mask'], surface['hue'])
            debug_analysis = overlay_mask(debug_analysis, surface['mask'], surface['hue'])

        surface["distance_factor"] = 1 / surface["rawParams"][1]
        pixelCount = cv2.countNonZero(surface['mask'])
        area = math.sqrt(pixelCount)
        max_erode = max(int(area/40), 2)

        surface['mask_expanded'] = rough_dilate_erode(True, surface['mask'], max_erode, scale=scale)
        markers[surface['mask'] > 0] = surface['label']
        outliers_mask[surface['mask_expanded'] > 0] = 0

    watershed_source = cv2.addWeighted(img, 0.7,  cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB), 0.333, 0)
    markers[outliers_mask>0] = WatershedLabel.OUTLIER_LABEL
    markers = Line.draw_all(line_data, markers, color=WatershedLabel.UNKNOWN_LABEL, thickness=int(diagonal/40))
    watershed_source = Line.draw_all(line_data, watershed_source, color=(0,255,0), thickness=2)
    
    if SAVE_DEBUG_IMAGES:
        debug_markers = translate_markers(watershed_source, markers, len(surfaces), hues)

    result = cv2.watershed(watershed_source, markers)

    if SAVE_DEBUG_IMAGES:
        debug_analysis = Line.draw_all(line_data, debug_analysis)
        debug_before = np.hstack((debug_before, debug_analysis))
        debug_final = translate_markers(img.copy(), result, len(surfaces), hues)
        debug_final = np.hstack((debug_markers, debug_final))
        cv2.imwrite(os.path.join(output_path, "debug.jpg"), np.vstack((debug_before, debug_final)))
        

def get_file_paths(input_dir, pattern):
    files = []
    if pattern:
        files.extend(Path(input_dir).glob('**/' + pattern))
    else: 
        extensions = ('.png', '.jpg', '.jpeg')
        for ext in extensions:
            files.extend(Path(input_dir).glob('**/*' + ext))
    return files

def parse_data(input_dir, output_dir):

    files = get_file_paths(input_dir, "**/data_v2.json")
    index = 0

    for path in files:
        data_path = str(path)
        dir_path = os.path.dirname(data_path)

        print("Parsing", path)
        with open(path) as f:
            data = json.load(f)
        
        image_path = os.path.join(dir_path, "background")

        if not os.path.exists(image_path):
            print("No image at path", image_path)
            continue

        surfaces = data['geometry']['surfaces']
        masks = []

        if len(surfaces) < 3:
            continue

        img = cv2.imread(image_path)

        hfov = data["camera"]["fov"]
        aspect = float(img.shape[1]) / float(img.shape[0])
        
        parts = dir_path.split(input_dir)
        if len(parts) > 1:
            file_name = parts[1][1:]
            output_path = os.path.join(output_dir, file_name)
        else:
            file_name = "image_" + str(index)
            output_path = output_dir

        surface_invalid = False

        #make output dir
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        #run segmentation:
        seg_path = os.path.join(output_path, "mask.png")

        #process each surface:
        for surface in surfaces:
            mask_url = surface['images']['mask']
            mask_path = os.path.join(dir_path, mask_url[mask_url.index('plane_masks'):])
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if mask is None:
                surface_invalid = True
                continue

            surface['mask'] = mask

        if surface_invalid:
            print("Invalid surface")
            continue

        find_surfaces(img, surfaces, output_path)

        index += 1
    return index

@click.command()
@click.argument("input_dir", default='input', type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.argument("output_dir", default='output', type=click.Path(exists=False, file_okay=False, dir_okay=True))
def main(input_dir, output_dir):

    if not os.path.exists(input_dir):
        raise Exception('The directory does not exist at path {}'.format(input_dir)) 

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    start = time.time()

    num_files = parse_data(input_dir, output_dir)
    
    elapsed = (time.time() - start)
    print("Processing %d images took %.2f seconds (%.2fs per image)" % (num_files, elapsed, elapsed/num_files))

if __name__ == "__main__":
    main()
