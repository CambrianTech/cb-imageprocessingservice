import math
import os
import json
import cv2
import numpy as np
from pathlib import Path

import click
import time

from scipy.special import softmax

import mxnet as mx
from mxnet.gluon.data.vision import transforms
import gluoncv
from gluoncv.utils.viz import get_color_pallete
import matplotlib.image as mpimg

import cambrian.image_processing as ip

import io
from cambrian import frei_chen, VanishingPointFinder, Line, RectangleFinder, SegmentationLabel, SegmentationSet
from modelutils import feed_image_batched, feed_images_batched, load_model
from gluoncv.data.transforms.presets.segmentation import test_transform
import random

#SEG_RES = None
#ADE_MODEL = 'deeplab_resnet101_ade'

ADE_MODEL = 'deeplab_resnest269_ade'
SEG_RES = 480

SAVE_DEBUG_IMAGES = True 

def segment_images(ctx, model, images, datum):
    # Numpy to mx, resize, test-transform, batch
    semantic_input = [
        test_transform(
            mx.img.resize_short(
                mx.nd.array(image, dtype=np.uint8),
                SEG_RES
            ),
            ctx
        )
        for image in images
    ]

    # Run semantic segmentation model
    # TODO: Batch properly?
    semantic_results = [
        model.predict(inp).asnumpy()
        for inp in semantic_input
    ]

    for result in semantic_results:
        datum["semantic"] = result
        datum["semantic_probs"] = softmax(result[0], axis=0)

def colorize_labels(image):
    return np.array(get_color_pallete(image, 'ade20k').convert('RGB'))

def run_watershed(img, mask, distance, num_labels=152, adjusted_labels=None):
    markers = np.zeros(img.shape[:2], dtype=np.int32)
    
    labels = []
    for i in range(0, num_labels):
        num_elements = (mask == i).sum()
        if num_elements > 50:
            isolated = np.zeros(mask.shape, dtype=np.uint8)
            isolated[mask == i] = 255 
            isolated = cv2.distanceTransform(isolated, cv2.DIST_L2, 5)
            _, isolated = cv2.threshold(isolated, distance * isolated.max(), i + 1, 0)
            markers[isolated > 0] = i + 1
            labels.append(i)

    markers = cv2.watershed(img, markers)
    markers = markers - 1
    #markers[markers<0] = 0
    return markers.astype(np.uint8), labels

def is_image_edge(point_a, point_b, shape, dist=10):
    max_0 = shape[0] - 1
    max_1 = shape[1] - 1
    return (abs(point_a[0]) <= dist and abs(point_b[0]) <= dist) \
        or (abs(point_a[0] - max_0) <= dist and abs(point_b[0] - max_0) <= dist) \
        or (abs(point_a[1]) <= dist and abs(point_b[1]) <= dist) \
        or (abs(point_a[1] - max_1) <= dist and abs(point_b[1] - max_1) <= dist)

def find_lines(images, output_path):

    height, width = images["image"].shape[:2]
    diagonal = np.hypot(width, height)
    print("image w,h", width, height)

    bw = cv2.cvtColor(images["image"], cv2.COLOR_BGR2GRAY)
    gabor_scale = 1500.0 / diagonal
    bw_res = cv2.resize(bw, (int(width * gabor_scale), int(height * gabor_scale)), cv2.INTER_CUBIC) if gabor_scale < 1.0 else bw

    def _gabor(theta, lambd, gamma = 0.0, psi = 0.0):
        ksize = lambd
        sigma = ksize * lambd
        result = cv2.filter2D(bw_res, cv2.CV_8UC1, cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, psi, ktype=cv2.CV_32F))
        return result

    v_gabor = _gabor(0, 7)
    h_gabor = _gabor(np.pi/2.0, 9)

    contours_src = cv2.addWeighted(v_gabor, 1.0, h_gabor, 1.0, 0)
    contours_src = cv2.resize(contours_src, (width, height), interpolation = cv2.INTER_CUBIC)

    def add_contour_lines(contours, min_confidence=None):
        epsilon = diagonal / 200.0
        min_length = diagonal / 30.0
        contour_group = 0
        for contour in contours:
            poly = cv2.approxPolyDP(contour, epsilon, False)

            for i in range(0, len(poly)-1):
                point_a = poly[i][0]
                point_b = poly[i+1][0]

                if not is_image_edge(point_a, point_b, (width, height), epsilon+1.0):
                    new_line = Line(point_a[0], point_a[1], point_b[0], point_b[1])
                    if new_line.length >= min_length:
                        line_data.append(new_line)
    
    edges = cv2.addWeighted(v_gabor, 3.0, h_gabor, 3.0, -20)
    edges = cv2.bilateralFilter(edges, 5, 5, 5)

    images["clean_edges"] = (frei_chen(bw) * 5.0).astype("uint8")

    contours_src = cv2.adaptiveThreshold(contours_src, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, int(diagonal / 50) * 2 + 1, -30)
    images["edges"] = ip.rough_dilate_erode(True, contours_src, 3, scale=400/diagonal, interpolation=cv2.INTER_AREA)
    images["segmented"] = cv2.resize(images["segmented"], (width, height), interpolation = cv2.INTER_NEAREST)

    # img_enhanced = cv2.addWeighted(images["image"], 1.0, cv2.cvtColor(images["clean_edges"], cv2.COLOR_GRAY2BGR), 30.0, 0)
    # mask, labels = run_watershed(img_enhanced, mask, distance=0.04)       

    if SAVE_DEBUG_IMAGES:
        cv2.imwrite(os.path.join(output_path, "original.jpg"), images["image"])
        #cv2.imwrite(os.path.join(output_path, "enhanced.jpg"), img_enhanced)
        
        #debug = cv2.addWeighted(images["image"], 0.5, colorize_labels(mask), 0.5, 0)
        #cv2.imwrite(os.path.join(output_path, "seg_adjusted.jpg"), debug)

    #find lines
    Line.prepare(images)
    line_data = []

    # for label in labels:
    #     isolated = np.zeros(mask.shape, dtype=np.uint8)
    #     isolated[mask == label] = 255
    #     contours, _ = cv2.findContours(isolated, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    #     add_contour_lines(contours)

    def _add_lines(fld, image, min_confidence=None):
        lines = fld.detect(image)
        if lines is None: return

        sy = height / image.shape[0]
        sx = width / image.shape[1]

        if sx != 1.0 or sy != 1.0:
            lines = lines * [[sx, sy, sx, sy]]

        for line in lines: 
            new_line = Line(line[0][0], line[0][1], line[0][2], line[0][3])
            if min_confidence is None or new_line.get_confidence() > min_confidence: 
                line_data.append(new_line)

    #find all lines in the edge image
    fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 30.0), 1.41, 200, 240, 3, False)
    _add_lines(fld, edges, 0.2)

    aperture = 5 if diagonal > 1200 else 3
    fld = cv2.ximgproc.createFastLineDetector(int(diagonal / 50.0), 1.41, 200, 220, aperture, False)
    _add_lines(fld, bw - (images["clean_edges"] * 5.0).astype("uint8"), 0.15)

    # fld = cv2.ximgproc.createFastLineDetector(64, _canny_aperture_size=7, _do_merge=False)
    # _add_lines(fld, cv2.cvtColor(images["normals"], cv2.COLOR_BGR2GRAY))

    if SAVE_DEBUG_IMAGES:
        lines_a = cv2.addWeighted(images["image"], 0.5, cv2.resize(images["normals"], (width, height)), 0.5, 0)
        lines_b = cv2.addWeighted(images["image"], 0.5, colorize_labels(images["segmented"]), 0.5, 0)
        lines_c = images["image"].copy()
        lines_d = images["image"].copy()
        intersections = []
    else:
        lines_a = lines_b = lines_c = lines_d = None
    
    if lines_a is not None: Line.draw_all(line_data, lines_a)
    
    line_data = Line.merge(line_data, diagonal / 300.0, search_length=1.1, angle_threshold=math.radians(5.0), max_color_std=7.0)

    if lines_b is not None: Line.draw_all(line_data, lines_b)

    line_data = Line.merge(line_data, diagonal / 80.0, search_length=1.0, angle_threshold=math.radians(3.0), create_pairs=True)

    if lines_c is not None: Line.draw_all(line_data, lines_c)

    line_data, intersections = Line.find_corners(line_data, search_length=1.5, angle_threshold=math.radians(10.0), parallel_threshold=math.radians(4), \
            max_color_std=3.0, confidence_diff=0.4)

    line_data = Line.merge(line_data, diagonal / 300.0, search_length=1.07, angle_threshold=math.radians(5.0))

    if lines_d is not None: Line.draw_all(line_data, lines_d)

    if lines_d is not None:
        for intersection in intersections: cv2.circle(lines_c, (int(intersection[0]), int(intersection[1])), max(int(diagonal/100), 3), (255,0,0), 2)
        
        cont_img = np.vstack((np.hstack((lines_a, lines_b)), np.hstack((lines_c, lines_d))))
        cv2.imwrite(os.path.join(output_path, "cont.jpg"), cont_img)

    return line_data

def find_surfaces(images, line_data, output_path):

    if SAVE_DEBUG_IMAGES:
        vp_image = images["image"].copy()
        surfaces_image = images["image"].copy()
    else:
        surfaces_image = vp_image = None

    #vanishing points:
    vpf = VanishingPointFinder(line_data)
    vanishing_points = vpf.compute()
    
    if SAVE_DEBUG_IMAGES and len(vanishing_points) > 0:
        for vp in vanishing_points:
            color = np.random.randint(0, 255, size=(3, ))
            color = ( int (color [ 0 ]), int (color [ 1 ]), int (color [ 2 ]))
            for line in vp.inliers: 
                line.draw(vp_image, color=color, thickness=3)

    rf = RectangleFinder(images, vanishing_points, debug=surfaces_image)
    rectangles = rf.compute()
    surfaces_image = rf.debug

    cv2.imwrite(os.path.join(output_path, "surfaces.jpg"), np.hstack((vp_image, surfaces_image)))

def get_file_paths(input_dir, pattern):
    files = []
    if pattern:
        files.extend(Path(input_dir).glob('**/' + pattern))
    else: 
        extensions = ('.png', '.jpg', '.jpeg')
        for ext in extensions:
            files.extend(Path(input_dir).glob('**/*' + ext))
    return files

def parse_data(input_dir, output_dir, model_normals):

    files = get_file_paths(input_dir, "**/data_v2.json")
    index = 0

    ctx = mx.cpu(0) if mx.context.num_gpus() == 0 else mx.gpu(0)
    model = gluoncv.model_zoo.get_model(ADE_MODEL, pretrained=True)

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
        normals_path = os.path.join(output_path, "normals.png")

        datum = {}
        datum['image'] = img
        datum['segmented'] = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)
        datum['normals'] = cv2.imread(normals_path)

        if datum['segmented'] is None:
            print("Segmenting %s" % image_path)
            start = time.process_time()
            segment_images(ctx, model, [img], datum)
            end = time.process_time()
            print("segmentation took %.2f seconds" % (end-start))
            datum['segmented'] = np.argmax(datum['semantic_probs'], axis=0)
            cv2.imwrite(seg_path, datum['segmented'])

        datum['labels'] = [SegmentationLabel(x) for x in list(np.unique(datum['segmented']))]

        print(datum['labels'])

        datum['segmented_color'] = colorize_labels(datum['segmented'])
        if SAVE_DEBUG_IMAGES:
            debug = cv2.addWeighted(datum["image"], 0.5, cv2.resize(datum["segmented_color"], (img.shape[1], img.shape[0])), 0.5, 0)
            cv2.imwrite(os.path.join(output_path, "seg_initial.jpg"), debug)

        if datum['normals'] is None:
            datum['normals'] = feed_image_batched(model_normals, [cv2.resize(img, (512, 512))])[0].astype("uint8")
            cv2.imwrite(normals_path, datum['normals'])

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

        line_data = find_lines(datum, output_path)

        if len(line_data) > 4:
            surfaces = find_surfaces(datum, line_data, output_path)

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

    normals_path = "models/normals"
    model_normals = load_model(normals_path)

    num_files = parse_data(input_dir, output_dir, model_normals)
    
    elapsed = (time.time() - start)
    print("Processing %d images took %.2f seconds (%.2fs per image)" % (num_files, elapsed, elapsed/num_files))

if __name__ == "__main__":
    main()
