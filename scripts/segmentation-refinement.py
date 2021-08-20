import math
import os
import json
import cv2
import numpy as np
from pathlib import Path

import click
import time

import io
from cambrian.frei_chen import frei_chen
from cambrian.Line import Line
from cambrian.VanishingPointFinder import VanishingPointFinder

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

def find_lines(img, output_path):

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

    contours_src = cv2.adaptiveThreshold(contours_src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, int(diagonal / 50) * 2 + 1, -30)
    contours_dilated = rough_dilate_erode(True, contours_src, 3, scale=400/diagonal, interpolation=cv2.INTER_AREA)
    Line.prepare(img, contours_dilated)

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
    
    line_data = Line.merge(line_data, diagonal / 300.0, search_length=1.1, angle_threshold=math.radians(5.0), max_color_std=7.0)

    if lines_b is not None: Line.draw_all(line_data, lines_b)

    line_data = Line.merge(line_data, diagonal / 80.0, search_length=1.0, angle_threshold=math.radians(5.0), create_pairs=True)

    if lines_c is not None: Line.draw_all(line_data, lines_c)

    line_data, intersections = Line.find_corners(line_data, search_length=1.5, angle_threshold=math.radians(10.0), parallel_threshold=math.radians(4), \
            max_color_std=3.0, confidence_diff=0.4)

    line_data = Line.merge(line_data, diagonal / 300.0, search_length=1.07, angle_threshold=math.radians(5.0))

    if lines_d is not None: Line.draw_all(line_data, lines_d)

    #vanishing points:
    vpf = VanishingPointFinder(line_data)
    vanishing_points = vpf.compute()
    
    if len(vanishing_points) > 0:
        vp = vanishing_points[0]
        inliers = np.array(line_data)[vp.votes > 0]
        for line in inliers:
            line.draw(lines_d, color=(255,0,255), thickness=3)

    if lines_d is not None:
        for intersection in intersections: cv2.circle(lines_c, (int(intersection[0]), int(intersection[1])), max(int(diagonal/100), 3), (255,0,0), 2)
        
        cont_img = np.vstack((np.hstack((lines_a, lines_b)), np.hstack((lines_c, lines_d))))
        cv2.imwrite(os.path.join(output_path, "cont.jpg"), cont_img)
        

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

        find_lines(img, output_path)

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
